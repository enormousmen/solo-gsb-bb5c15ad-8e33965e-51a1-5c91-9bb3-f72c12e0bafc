"""预约候补队列核心逻辑：可约则直接成约，不可约则按 FIFO 排队，取消时原子递补。

候补队列是正式预约的“排队等待”通道：

- ``POST /api/v1/reservation-waitlist`` 在单个 ``BEGIN IMMEDIATE`` 事务内：
  先把已到期的 pending 临时保留**惰性标记为 expired**，再按与正式预约创建
  完全相同的规则检查**进行中预约**（``reservation_participants``）与
  **未过期 pending 保留**（``reservation_hold_participants``）的半开区间
  ``[start, end)`` 参与者重叠——首尾相接不算冲突：

  - 时段当前可预约（无任何冲突）→ **同事务直接创建正式预约**并追加
    ``created`` 哈希链审计事件，候补行标记为 ``promoted`` 并回填
    ``reservation_id`` / ``promoted_at``（占用语义由正式预约接管，
    候补参与者占用行删除）；
  - 否则把候补写入 ``pending`` 队列，**按 ``created_at``（同刻按 id）
    决定 FIFO 顺序**；pending 候补本身**不占用时段**，不阻止任何
    正式预约、改期或临时保留。

- 相同 ``request_id`` 幂等：同键 + 相同规范化内容（起止 + 去重排序
  参与者）重试 → 返回原候补行（pending 返回 ``200``、已 promoted
  返回 ``200`` 并带关联预约）；同键内容不同 → :class:`WaitlistRequestConflictError`。
  ``request_id`` 已被正式预约占用同样按 409 处理（与临时保留确认一致）。

- 取消正式预约时（见 :func:`app.reservations.cancel_reservation`），在
  **同一个** ``BEGIN IMMEDIATE`` 事务内调用 :func:`promote_affected_waitlist`：
  按 FIFO 扫描“受影响候补”（至少一位参与者同时出现在被取消预约中、
  且窗口与被取消窗口半开重叠），**跳过仍冲突项**，把首个可满足项原子
  转为正式预约并追加 ``created`` 审计事件；至多递补一项，未中选候补
  保持 pending 不变。幂等重试取消不会再次触发递补。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from . import db
from . import holds as hold_core
from .reservations import (
    RequestIdConflictError,
    Reservation,
    _canon,
    _parse,
    collect_reservation_conflicts,
    insert_reservation,
    load_reservation,
    normalize_participants,
)

_WAITLIST_COLUMNS = (
    "id, request_id, start_utc, end_utc, participants_key, status,"
    " reservation_id, created_at, promoted_at"
)
_WAITLIST_COLUMNS_PREFIXED = ", ".join(
    f"w.{c}" for c in _WAITLIST_COLUMNS.split(", "))


class WaitlistNotFoundError(Exception):
    """指定 request_id 的候补不存在。"""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        super().__init__(f"reservation waitlist entry {request_id!r} not found")


class WaitlistRequestConflictError(Exception):
    """request_id 已被内容不同的候补占用。"""

    def __init__(self, request_id: str, waitlist_id: int) -> None:
        self.request_id = request_id
        self.waitlist_id = waitlist_id
        super().__init__(
            f"request_id is already used by waitlist entry #{waitlist_id} "
            "with different content"
        )


@dataclass(frozen=True)
class WaitlistEntry:
    id: int
    request_id: str
    start_utc: datetime
    end_utc: datetime
    participants: tuple[str, ...]
    created_at: datetime
    status: str = "pending"  # pending / promoted
    reservation_id: int | None = None
    promoted_at: datetime | None = None
    # 非持久化字段：promoted 时关联的正式预约（同事务内组装或读取路径加载）
    reservation: Reservation | None = None


def _row_to_entry(row: sqlite3.Row,
                  reservation: Reservation | None = None) -> WaitlistEntry:
    return WaitlistEntry(
        id=row["id"],
        request_id=row["request_id"],
        start_utc=_parse(row["start_utc"]),
        end_utc=_parse(row["end_utc"]),
        participants=tuple(json.loads(row["participants_key"])),
        status=row["status"],
        reservation_id=row["reservation_id"],
        created_at=_parse(row["created_at"]),
        promoted_at=_parse(row["promoted_at"]) if row["promoted_at"] else None,
        reservation=reservation,
    )


def _load_entry_by_request_id(
    conn: sqlite3.Connection, request_id: str
) -> WaitlistEntry | None:
    """在调用方连接/事务内按 request_id 读取候补；promoted 时一并加载关联预约。"""
    row = conn.execute(
        f"SELECT {_WAITLIST_COLUMNS} FROM reservation_waitlist"
        " WHERE request_id = ?",
        (request_id,),
    ).fetchone()
    if row is None:
        return None
    reservation = None
    if row["status"] == "promoted" and row["reservation_id"] is not None:
        reservation = load_reservation(conn, row["reservation_id"])
    return _row_to_entry(row, reservation)


def _collect_occupancy_conflicts(
    conn: sqlite3.Connection,
    participants: tuple[str, ...],
    start_s: str,
    end_s: str,
    now_s: str,
) -> list[dict[str, object]]:
    """与正式预约创建同口径的占用冲突：进行中预约 + 未过期 pending 保留。

    pending 候补不参与冲突（不占用时段）。
    """
    conflicts = collect_reservation_conflicts(conn, participants, start_s, end_s)
    conflicts.extend(
        hold_core.collect_hold_conflicts(conn, participants, start_s, end_s, now_s))
    return conflicts


def _promote_entry(
    conn: sqlite3.Connection,
    *,
    waitlist_id: int,
    request_id: str,
    start_s: str,
    end_s: str,
    participants: tuple[str, ...],
    now_s: str,
) -> int:
    """在调用方事务内把一条候补原子转为正式预约，返回新预约 ID。

    正式预约写入（含 ``created`` 审计事件）、候补状态翻转与候补占用行删除
    同事务提交。入队直接成约（内容已知）与取消递补（内容来自候补行）共用。
    """
    reservation_id = insert_reservation(
        conn,
        request_id=request_id,
        start_s=start_s,
        end_s=end_s,
        participants=participants,
        now_s=now_s,
    )
    conn.execute(
        "UPDATE reservation_waitlist"
        " SET status = 'promoted', reservation_id = ?, promoted_at = ?"
        " WHERE id = ?",
        (reservation_id, now_s, waitlist_id),
    )
    # 占用已由正式预约接管：删除候补参与者行，promoted 行借此退出 FIFO 扫描
    conn.execute(
        "DELETE FROM reservation_waitlist_participants WHERE waitlist_id = ?",
        (waitlist_id,),
    )
    return reservation_id


def promote_affected_waitlist(
    conn: sqlite3.Connection,
    *,
    canceled_start_s: str,
    canceled_end_s: str,
    canceled_participants: tuple[str, ...],
    now_s: str,
) -> int | None:
    """取消事务内按 FIFO 扫描受影响候补，递补首个可满足项。

    “受影响候补”指至少一位参与者同时出现在被取消预约中、且该参与者窗口与
    被取消窗口半开重叠的 pending 候选项；按 ``created_at ASC, id ASC``
    （严格 FIFO）逐条检查：

    - ``request_id`` 已被正式预约占用（理论上不该发生的防御性跳过，
      例如同一 request_id 曾被其他通道直接用于创建）或仍与**任何**
      进行中预约 / 未过期 pending 保留冲突 → **跳过该项，继续扫描**；
    - 首个完全无冲突的候选项 → :func:`_promote_entry` 原子转正式预约
      （同事务写 ``created`` 审计事件），返回其预约 ID；
    - 没有任何可满足项 → 返回 ``None``，所有候补保持原状。

    本函数在调用方已打开的写事务内执行，不自行提交；至多递补一项。
    """
    if not canceled_participants:
        return None
    placeholders = ",".join("?" for _ in canceled_participants)
    candidate_rows = conn.execute(
        f"SELECT DISTINCT {_WAITLIST_COLUMNS_PREFIXED}"
        " FROM reservation_waitlist w"
        " JOIN reservation_waitlist_participants wp ON wp.waitlist_id = w.id"
        f" WHERE w.status = 'pending'"
        f"   AND wp.participant_id IN ({placeholders})"
        "   AND wp.start_utc < ? AND ? < wp.end_utc"
        " ORDER BY w.created_at ASC, w.id ASC",
        (*canceled_participants, canceled_end_s, canceled_start_s),
    ).fetchall()

    for row in candidate_rows:
        # 防御性跳过：request_id 已被正式预约占用则无法再以该键成约
        existing = conn.execute(
            "SELECT 1 FROM reservations WHERE request_id = ?",
            (row["request_id"],),
        ).fetchone()
        if existing is not None:
            continue
        participants = tuple(json.loads(row["participants_key"]))
        if _collect_occupancy_conflicts(
                conn, participants, row["start_utc"], row["end_utc"], now_s):
            # 仍冲突（挡住它的不是刚取消的这场，或它还在等其他参与者）→ 跳过
            continue
        return _promote_entry(
            conn,
            waitlist_id=row["id"],
            request_id=row["request_id"],
            start_s=row["start_utc"],
            end_s=row["end_utc"],
            participants=participants,
            now_s=now_s,
        )
    return None


def join_waitlist(
    request_id: str,
    start_utc: datetime,
    end_utc: datetime,
    participant_ids: Iterable[str],
    db_path: str | None = None,
) -> tuple[WaitlistEntry, bool]:
    """加入候补或直接成约，返回 ``(候补行, 是否新建)``。

    整个流程在单个 ``BEGIN IMMEDIATE`` 事务内完成，任何一步失败都回滚：

    1. 惰性过期到期临时保留；
    2. 同一 ``request_id`` 已有候补：规范化内容相同 → 幂等返回原行
       （``created=False``，状态可能是 pending 或 promoted）；
       内容不同 → :class:`WaitlistRequestConflictError`（409）；
    3. ``request_id`` 已被正式预约占用 → :class:`RequestIdConflictError`（409）；
    4. 与进行中预约 / 未过期 pending 保留无冲突 → 直接创建正式预约：
       写候补行、正式预约与 ``created`` 审计事件，候补置为 promoted；
    5. 有冲突 → 候补行置为 pending 排队（FIFO 由 created_at, id 决定），
       不写入任何正式预约占用。
    """
    participants = normalize_participants(participant_ids)
    start_s = _canon(start_utc)
    end_s = _canon(end_utc)
    participants_key = json.dumps(list(participants), ensure_ascii=False)

    path = db.ensure_initialized(db_path)
    conn = db.connect(path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            now_s = _canon(datetime.now(timezone.utc))
            hold_core.expire_due_holds(conn, now_s)

            existing_row = conn.execute(
                f"SELECT {_WAITLIST_COLUMNS} FROM reservation_waitlist"
                " WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if existing_row is not None:
                if (existing_row["start_utc"] == start_s
                        and existing_row["end_utc"] == end_s
                        and existing_row["participants_key"] == participants_key):
                    conn.execute("COMMIT")
                    return _load_entry_by_request_id(conn, request_id), False
                raise WaitlistRequestConflictError(
                    request_id, existing_row["id"])

            # request_id 已被正式预约占用：复用预约体系的 409 语义
            reservation_row = conn.execute(
                f"SELECT id FROM reservations WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if reservation_row is not None:
                raise RequestIdConflictError(reservation_row["id"])

            cursor = conn.execute(
                "INSERT INTO reservation_waitlist"
                " (request_id, start_utc, end_utc, participants_key,"
                "  status, created_at)"
                " VALUES (?, ?, ?, ?, 'pending', ?)",
                (request_id, start_s, end_s, participants_key, now_s),
            )
            waitlist_id = int(cursor.lastrowid)
            conn.executemany(
                "INSERT INTO reservation_waitlist_participants"
                " (waitlist_id, participant_id, start_utc, end_utc)"
                " VALUES (?, ?, ?, ?)",
                [(waitlist_id, pid, start_s, end_s) for pid in participants],
            )

            conflicts = _collect_occupancy_conflicts(
                conn, participants, start_s, end_s, now_s)
            if conflicts:
                # 当前不可预约：保持 pending，按创建时间进入 FIFO 队列
                conn.execute("COMMIT")
                return WaitlistEntry(
                    id=waitlist_id,
                    request_id=request_id,
                    start_utc=_parse(start_s),
                    end_utc=_parse(end_s),
                    participants=participants,
                    created_at=_parse(now_s),
                    status="pending",
                ), True

            # 当前可预约：同事务直接转正式预约（含 created 审计事件）
            reservation_id = _promote_entry(
                conn,
                waitlist_id=waitlist_id,
                request_id=request_id,
                start_s=start_s,
                end_s=end_s,
                participants=participants,
                now_s=now_s,
            )
            conn.execute("COMMIT")
            return WaitlistEntry(
                id=waitlist_id,
                request_id=request_id,
                start_utc=_parse(start_s),
                end_utc=_parse(end_s),
                participants=participants,
                created_at=_parse(now_s),
                status="promoted",
                reservation_id=reservation_id,
                promoted_at=_parse(now_s),
                reservation=Reservation(
                    id=reservation_id,
                    request_id=request_id,
                    start_utc=_parse(start_s),
                    end_utc=_parse(end_s),
                    participants=participants,
                    created_at=_parse(now_s),
                    version=1,
                ),
            ), True
        except BaseException:
            conn.rollback()
            raise
    finally:
        conn.close()


def get_waitlist_entry(
    request_id: str,
    db_path: str | None = None,
) -> WaitlistEntry:
    """按 request_id 读取候补状态；不存在 → :class:`WaitlistNotFoundError`。

    promoted 时一并加载关联正式预约，供 API 返回 ``reservation``。
    """
    path = db.ensure_initialized(db_path)
    conn = db.connect(path)
    try:
        entry = _load_entry_by_request_id(conn, request_id)
        if entry is None:
            raise WaitlistNotFoundError(request_id)
        return entry
    finally:
        conn.close()
