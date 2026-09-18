"""预约候补队列（reservation waitlist）测试。

覆盖：
- POST：时段可约直接成约 201（promoted + created 审计事件）、冲突排队 202、
  幂等重放 200（pending/promoted）、同 request_id 异内容 409、
  request_id 已被正式预约占用 409、参数校验 422、GET 404；
- pending 候补不占位（不阻止正式预约/改期/临时保留）；
- 取消事务内 FIFO 递补：首个可满足项原子转预约、跳过仍冲突项、
  只扫描受影响候补、幂等取消重试不二次递补、版本不符 409 不递补、
  到期保留在同一事务惰性过期后放行、链性递补、同事务哈希链可验；
- 并发：同 request_id 并发只产生一条候补/预约；取消递补与直接抢占
  在 BEGIN IMMEDIATE 串行化下绝不双重占用；
- 迁移：旧版库无损建立候补表，幂等可重复执行。
"""

from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app import db as db_module
from app import reservations as reservation_core
from app import waitlist as waitlist_core
from app.main import app


@pytest.fixture()
def db_path(tmp_path, monkeypatch) -> str:
    """每个测试用独立的 SQLite 文件。"""
    path = str(tmp_path / "reservations.db")
    monkeypatch.setenv("RESERVATIONS_DB_PATH", path)
    return path


@pytest.fixture()
def client(db_path) -> TestClient:
    return TestClient(app)


def _wl_payload(**overrides):
    payload = {
        "request_id": "wl-1",
        "start_utc": "2026-03-09T13:00:00Z",
        "end_utc": "2026-03-09T14:00:00Z",
        "participant_ids": ["alice", "bob"],
    }
    payload.update(overrides)
    return payload


def _resv_payload(**overrides):
    payload = {
        "request_id": "req-1",
        "start_utc": "2026-03-09T13:00:00Z",
        "end_utc": "2026-03-09T14:00:00Z",
        "participant_ids": ["alice", "bob"],
    }
    payload.update(overrides)
    return payload


def _hold_payload(**overrides):
    payload = {
        "hold_key": "hold-1",
        "start_utc": "2026-03-09T13:00:00Z",
        "end_utc": "2026-03-09T14:00:00Z",
        "participant_ids": ["alice"],
        "ttl_seconds": 300,
    }
    payload.update(overrides)
    return payload


def _utc(y, mo, d, h, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc)


def _force_hold_expired(path: str, hold_id: int) -> None:
    """直接把保留的 expires_at 改到过去（不翻转 status），模拟 TTL 到期。"""
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "UPDATE reservation_holds SET expires_at = ? WHERE id = ?",
            ("2000-01-01T00:00:00.000000Z", hold_id),
        )
        conn.commit()
    finally:
        conn.close()


def _cancel(client, reservation_id: int, expected_version: int = 1):
    return client.delete(
        f"/api/v1/reservations/{reservation_id}",
        params={"expected_version": expected_version},
    )


# --------------------------------------------------------------- POST：直接成约

def test_join_free_slot_creates_reservation_201(client):
    resp = client.post("/api/v1/reservation-waitlist", json=_wl_payload())
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["waitlist_id"] == 1
    assert data["request_id"] == "wl-1"
    assert data["status"] == "promoted"
    assert data["promoted_at"] is not None
    assert data["participants"] == ["alice", "bob"]
    reservation = data["reservation"]
    assert reservation is not None
    assert reservation["request_id"] == "wl-1"
    assert reservation["start_utc"] == "2026-03-09T13:00:00Z"
    assert reservation["end_utc"] == "2026-03-09T14:00:00Z"
    assert reservation["status"] == "active"
    assert reservation["version"] == 1


def test_direct_promotion_normalizes_participants(client):
    resp = client.post("/api/v1/reservation-waitlist",
                       json=_wl_payload(participant_ids=["bob", "alice", "bob"]))
    assert resp.status_code == 201, resp.text
    assert resp.json()["participants"] == ["alice", "bob"]
    assert resp.json()["reservation"]["participants"] == ["alice", "bob"]


def test_direct_promotion_occupies_slot_and_has_created_event(client):
    promoted = client.post("/api/v1/reservation-waitlist",
                           json=_wl_payload()).json()
    rid = promoted["reservation"]["reservation_id"]

    # 正式预约实际占用时段：他人同时段冲突
    blocked = client.post("/api/v1/reservations", json=_resv_payload(
        request_id="req-other", participant_ids=["alice"],
        start_utc="2026-03-09T13:30:00Z", end_utc="2026-03-09T13:45:00Z"))
    assert blocked.status_code == 409
    assert blocked.json()["conflicts"] == [
        {"reservation_id": rid, "participant_id": "alice"}]

    # 同事务写了 created 哈希链事件，链可验
    history = client.get(f"/api/v1/reservations/{rid}/history").json()
    assert [e["action"] for e in history["events"]] == ["created"]


def test_direct_promotion_deletes_waitlist_occupancy_rows(client, db_path):
    promoted = client.post("/api/v1/reservation-waitlist",
                           json=_wl_payload()).json()
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT status, reservation_id, promoted_at FROM reservation_waitlist"
            " WHERE id = ?", (promoted["waitlist_id"],)).fetchone()
        assert row[0] == "promoted"
        assert row[1] == promoted["reservation"]["reservation_id"]
        assert row[2] is not None
        # 候补占用行已删除（占用由正式预约接管）
        assert conn.execute(
            "SELECT COUNT(*) FROM reservation_waitlist_participants"
        ).fetchone()[0] == 0
    finally:
        conn.close()


# --------------------------------------------------------------- POST：排队

def test_conflicting_join_queues_202(client):
    assert client.post("/api/v1/reservations", json=_resv_payload(
        participant_ids=["alice"])).status_code == 201
    resp = client.post("/api/v1/reservation-waitlist", json=_wl_payload())
    assert resp.status_code == 202, resp.text
    data = resp.json()
    assert data["status"] == "pending"
    assert data["reservation"] is None
    assert data["promoted_at"] is None
    assert data["created_at"]


def test_adjacent_window_is_bookable_not_queued(client):
    # 半开区间相邻：13:00–14:00 已约，14:00–15:00 仍可直接成约
    assert client.post("/api/v1/reservations", json=_resv_payload(
        participant_ids=["alice"])).status_code == 201
    resp = client.post("/api/v1/reservation-waitlist", json=_wl_payload(
        request_id="wl-2",
        start_utc="2026-03-09T14:00:00Z",
        end_utc="2026-03-09T15:00:00Z"))
    assert resp.status_code == 201, resp.text
    assert resp.json()["status"] == "promoted"


def test_hold_conflict_queues(client):
    assert client.post("/api/v1/reservation-holds",
                       json=_hold_payload()).status_code == 201
    resp = client.post("/api/v1/reservation-waitlist", json=_wl_payload())
    assert resp.status_code == 202, resp.text
    assert resp.json()["status"] == "pending"


def test_pending_entry_does_not_block_reservation(client):
    # alice 13–14 的正式预约挡住 alice 13:30 的候补
    assert client.post("/api/v1/reservations", json=_resv_payload(
        participant_ids=["alice"])).status_code == 201
    assert client.post("/api/v1/reservation-waitlist", json=_wl_payload(
        request_id="wl-alice",
        start_utc="2026-03-09T13:30:00Z",
        end_utc="2026-03-09T13:45:00Z",
        participant_ids=["alice"],
    )).status_code == 202
    # 候补不占位：dave 同窗口直接预约不受候补影响（不相交参与者）
    resp = client.post("/api/v1/reservations", json=_resv_payload(
        request_id="req-dave",
        start_utc="2026-03-09T13:30:00Z",
        end_utc="2026-03-09T13:45:00Z",
        participant_ids=["dave"],
    ))
    assert resp.status_code == 201, resp.text


def test_pending_entry_does_not_block_hold(client):
    # carol 13–14 挡住 carol 13:30 的候补
    assert client.post("/api/v1/reservations", json=_resv_payload(
        participant_ids=["carol"])).status_code == 201
    assert client.post("/api/v1/reservation-waitlist", json=_wl_payload(
        request_id="wl-carol",
        participant_ids=["carol"],
        start_utc="2026-03-09T13:30:00Z",
        end_utc="2026-03-09T13:45:00Z",
    )).status_code == 202
    # bob 在同一窗口的保留不与 carol 的候补冲突（候补不参与保留冲突检查）
    resp = client.post("/api/v1/reservation-holds", json=_hold_payload(
        hold_key="h-bob",
        start_utc="2026-03-09T13:30:00Z",
        end_utc="2026-03-09T13:45:00Z",
        participant_ids=["bob"],
    ))
    assert resp.status_code == 201, resp.text


def test_pending_entry_does_not_block_reschedule(client):
    # alice 的保留挡住 alice 13:30 的候补；另一场 alice 预约在次日 15–16
    assert client.post("/api/v1/reservation-holds", json=_hold_payload(
        hold_key="h-block")).status_code == 201
    r1 = client.post("/api/v1/reservations", json=_resv_payload(
        request_id="req-B",
        start_utc="2026-03-10T15:00:00Z",
        end_utc="2026-03-10T16:00:00Z",
        participant_ids=["alice"],
    )).json()
    assert client.post("/api/v1/reservation-waitlist", json=_wl_payload(
        request_id="wl-x",
        start_utc="2026-03-09T13:30:00Z",
        end_utc="2026-03-09T13:45:00Z",
        participant_ids=["alice"],
    )).status_code == 202
    # 候补不阻止把另一场改到同日保留之后（14:30，与保留半开相邻）
    resp = client.put(f"/api/v1/reservations/{r1['reservation_id']}", json={
        "expected_version": 1,
        "start_utc": "2026-03-09T14:30:00Z",
        "end_utc": "2026-03-09T14:45:00Z",
        "participant_ids": ["alice"],
    })
    assert resp.status_code == 200, resp.text
    # 候补仍 pending（改期不会触发递补）
    got = client.get("/api/v1/reservation-waitlist/wl-x").json()
    assert got["status"] == "pending"


# --------------------------------------------------------------- POST：幂等/冲突

def test_pending_idempotent_replay_returns_200_same_body(client):
    assert client.post("/api/v1/reservations", json=_resv_payload(
        participant_ids=["alice"])).status_code == 201
    first = client.post("/api/v1/reservation-waitlist", json=_wl_payload())
    assert first.status_code == 202
    replay = client.post("/api/v1/reservation-waitlist",
                         json=_wl_payload(participant_ids=["bob", "alice"]))
    assert replay.status_code == 200
    assert replay.json() == first.json()


def test_promoted_idempotent_replay_returns_200_with_reservation(client):
    first = client.post("/api/v1/reservation-waitlist", json=_wl_payload())
    assert first.status_code == 201
    replay = client.post("/api/v1/reservation-waitlist", json=_wl_payload())
    assert replay.status_code == 200
    assert replay.json() == first.json()
    assert replay.json()["status"] == "promoted"


def test_same_request_id_different_content_returns_409(client):
    assert client.post("/api/v1/reservation-waitlist",
                       json=_wl_payload()).status_code == 201
    resp = client.post("/api/v1/reservation-waitlist",
                       json=_wl_payload(end_utc="2026-03-09T15:00:00Z"))
    assert resp.status_code == 409
    assert resp.json()["existing_waitlist_id"] == 1
    resp = client.post("/api/v1/reservation-waitlist",
                       json=_wl_payload(participant_ids=["alice"]))
    assert resp.status_code == 409
    assert resp.json()["existing_waitlist_id"] == 1


def test_request_id_used_by_reservation_returns_409(client):
    other = client.post("/api/v1/reservations", json=_resv_payload(
        request_id="dup-req",
        start_utc="2026-04-01T13:00:00Z",
        end_utc="2026-04-01T14:00:00Z",
        participant_ids=["zed"],
    ))
    assert other.status_code == 201
    resp = client.post("/api/v1/reservation-waitlist",
                       json=_wl_payload(request_id="dup-req"))
    assert resp.status_code == 409
    assert resp.json()["existing_reservation_id"] == other.json()["reservation_id"]


def test_pending_waitlist_request_id_rejected_by_direct_create(client):
    # 候补排队中的 request_id 不允许被直接创建路径另作他用
    assert client.post("/api/v1/reservations", json=_resv_payload(
        participant_ids=["alice"])).status_code == 201
    assert client.post("/api/v1/reservation-waitlist",
                       json=_wl_payload(request_id="wl-pending")).status_code == 202
    resp = client.post("/api/v1/reservations", json=_resv_payload(
        request_id="wl-pending",
        start_utc="2026-04-01T13:00:00Z",
        end_utc="2026-04-01T14:00:00Z",
        participant_ids=["zed"],
    ))
    assert resp.status_code == 409


def test_promotion_defensively_skips_request_id_taken_reservation(client, db_path):
    """旁路（核心层/旧数据）造成候补 request_id 已被预约占用时，取消安全跳过它。"""
    blocked_by = client.post("/api/v1/reservations", json=_resv_payload(
        request_id="req-block", participant_ids=["alice"])).json()
    assert client.post("/api/v1/reservation-waitlist", json=_wl_payload(
        request_id="wl-zombie",
        start_utc="2026-03-09T13:30:00Z",
        end_utc="2026-03-09T13:45:00Z",
        participant_ids=["alice"],
    )).status_code == 202

    # 旁路地把同一 request_id 直接写入 reservations（绕过 API 守卫，模拟旧数据）
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO reservations"
            " (request_id, start_utc, end_utc, participants_key, version, created_at)"
            " VALUES ('wl-zombie', '2026-04-01T13:00:00.000000Z',"
            " '2026-04-01T14:00:00.000000Z', '[\"zed\"]', 1,"
            " '2026-03-01T00:00:00.000000Z')")
        conn.commit()
    finally:
        conn.close()

    # 取消不报错：该候补被防御性跳过，保持 pending
    assert _cancel(client, blocked_by["reservation_id"]).status_code == 200
    got = client.get("/api/v1/reservation-waitlist/wl-zombie").json()
    assert got["status"] == "pending"


def test_validation_errors_return_422(client):
    assert client.post("/api/v1/reservation-waitlist",
                       json=_wl_payload(request_id="")).status_code == 422
    assert client.post("/api/v1/reservation-waitlist",
                       json=_wl_payload(end_utc="2026-03-09T13:00:00Z")
                       ).status_code == 422
    assert client.post("/api/v1/reservation-waitlist",
                       json=_wl_payload(start_utc="2026-03-09T13:00:00")
                       ).status_code == 422
    assert client.post("/api/v1/reservation-waitlist",
                       json=_wl_payload(participant_ids=[])).status_code == 422


# ------------------------------------------------------------------- GET 状态

def test_get_pending_entry(client):
    assert client.post("/api/v1/reservations", json=_resv_payload(
        participant_ids=["alice"])).status_code == 201
    assert client.post("/api/v1/reservation-waitlist",
                       json=_wl_payload()).status_code == 202
    resp = client.get("/api/v1/reservation-waitlist/wl-1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "pending"
    assert data["reservation"] is None
    assert data["promoted_at"] is None


def test_get_unknown_request_id_returns_404(client):
    resp = client.get("/api/v1/reservation-waitlist/nope")
    assert resp.status_code == 404
    assert "nope" in resp.json()["detail"]


# --------------------------------------------------- 取消：FIFO 递补

def test_cancel_promotes_fifo_head_atomically(client):
    blocked_by = client.post("/api/v1/reservations", json=_resv_payload(
        request_id="req-block", participant_ids=["alice"])).json()
    w1 = client.post("/api/v1/reservation-waitlist", json=_wl_payload(
        request_id="wl-1",
        start_utc="2026-03-09T13:30:00Z",
        end_utc="2026-03-09T13:45:00Z",
        participant_ids=["alice"],
    ))
    assert w1.status_code == 202

    canceled = _cancel(client, blocked_by["reservation_id"], expected_version=1)
    assert canceled.status_code == 200, canceled.text

    got = client.get("/api/v1/reservation-waitlist/wl-1").json()
    assert got["status"] == "promoted"
    reservation = got["reservation"]
    assert reservation is not None
    assert reservation["start_utc"] == "2026-03-09T13:30:00Z"
    assert reservation["end_utc"] == "2026-03-09T13:45:00Z"
    assert reservation["participants"] == ["alice"]
    assert reservation["status"] == "active"
    assert got["promoted_at"] is not None

    # 递补的预约真实占用时段
    blocked = client.post("/api/v1/reservations", json=_resv_payload(
        request_id="req-late",
        start_utc="2026-03-09T13:30:00Z",
        end_utc="2026-03-09T13:45:00Z",
        participant_ids=["alice"],
    ))
    assert blocked.status_code == 409
    assert blocked.json()["conflicts"] == [
        {"reservation_id": reservation["reservation_id"], "participant_id": "alice"}]

    # 递补预约在同一事务写了 created 事件
    history = client.get(
        f"/api/v1/reservations/{reservation['reservation_id']}/history").json()
    assert [e["action"] for e in history["events"]] == ["created"]
    # 被取消预约的 canceled 事件同样在链上
    cancel_history = client.get(
        f"/api/v1/reservations/{blocked_by['reservation_id']}/history").json()
    assert [e["action"] for e in cancel_history["events"]][-1] == "canceled"


def test_cancel_fifo_order_earliest_wins_and_later_waits(client, db_path):
    blocked_by = client.post("/api/v1/reservations", json=_resv_payload(
        request_id="req-block", participant_ids=["alice"])).json()
    assert client.post("/api/v1/reservation-waitlist", json=_wl_payload(
        request_id="wl-earlier",
        start_utc="2026-03-09T13:30:00Z",
        end_utc="2026-03-09T13:45:00Z",
        participant_ids=["alice"],
    )).status_code == 202
    later = client.post("/api/v1/reservation-waitlist", json=_wl_payload(
        request_id="wl-later",
        start_utc="2026-03-09T13:30:00Z",
        end_utc="2026-03-09T13:45:00Z",
        participant_ids=["alice"],
    ))
    assert later.status_code == 202

    assert _cancel(client, blocked_by["reservation_id"]).status_code == 200

    first = client.get("/api/v1/reservation-waitlist/wl-earlier").json()
    second = client.get("/api/v1/reservation-waitlist/wl-later").json()
    assert first["status"] == "promoted"
    assert second["status"] == "pending"  # 被刚递补的预约挡住，继续等待

    conn = sqlite3.connect(db_path)
    try:
        # 只有被递补那一项的候补占用行被删除；pending 行保留
        remaining = conn.execute(
            "SELECT w.request_id FROM reservation_waitlist_participants wp"
            " JOIN reservation_waitlist w ON w.id = wp.waitlist_id"
        ).fetchall()
        assert remaining == [("wl-later",)]
    finally:
        conn.close()

    # 幂等重试取消（原 expected_version=1）不再触发第二次递补
    retry = _cancel(client, blocked_by["reservation_id"], expected_version=1)
    assert retry.status_code == 200
    assert client.get("/api/v1/reservation-waitlist/wl-later").json()["status"] == "pending"

    # 取消递补产生的预约 → 链式递补到 wl-later
    assert _cancel(
        client, first["reservation"]["reservation_id"], expected_version=1
    ).status_code == 200
    assert client.get("/api/v1/reservation-waitlist/wl-later").json()["status"] == "promoted"


def test_cancel_skips_still_conflicting_entry(client):
    # alice 的场次将被取消；bob 的同窗口场次保持进行中
    alice_slot = client.post("/api/v1/reservations", json=_resv_payload(
        request_id="req-alice", participant_ids=["alice"])).json()
    assert client.post("/api/v1/reservations", json=_resv_payload(
        request_id="req-bob", participant_ids=["bob"])).status_code == 201

    # W1 排在前面但需要 alice+bob：alice 释放后 bob 仍冲突 → 跳过
    assert client.post("/api/v1/reservation-waitlist", json=_wl_payload(
        request_id="wl-both",
        start_utc="2026-03-09T13:30:00Z",
        end_utc="2026-03-09T13:45:00Z",
        participant_ids=["alice", "bob"],
    )).status_code == 202
    # W2 只需要 alice → 首个可满足项，递补
    assert client.post("/api/v1/reservation-waitlist", json=_wl_payload(
        request_id="wl-alice-only",
        start_utc="2026-03-09T13:30:00Z",
        end_utc="2026-03-09T13:45:00Z",
        participant_ids=["alice"],
    )).status_code == 202

    assert _cancel(client, alice_slot["reservation_id"]).status_code == 200

    assert client.get("/api/v1/reservation-waitlist/wl-both").json()["status"] == "pending"
    promoted = client.get("/api/v1/reservation-waitlist/wl-alice-only").json()
    assert promoted["status"] == "promoted"
    assert promoted["reservation"]["participants"] == ["alice"]


def test_cancel_does_not_promote_unaffected_entries(client):
    """不同参与者/不重叠窗口的候补不受无关取消影响。"""
    alice_slot = client.post("/api/v1/reservations", json=_resv_payload(
        request_id="req-alice",
        start_utc="2026-03-09T13:00:00Z",
        end_utc="2026-03-09T14:00:00Z",
        participant_ids=["alice"],
    )).json()
    # dave 16–17 的另一场挡住 carol+dave 的候补
    assert client.post("/api/v1/reservations", json=_resv_payload(
        request_id="req-dave",
        start_utc="2026-03-09T16:00:00Z",
        end_utc="2026-03-09T17:00:00Z",
        participant_ids=["dave"],
    )).status_code == 201
    assert client.post("/api/v1/reservation-waitlist", json=_wl_payload(
        request_id="wl-other",
        start_utc="2026-03-09T16:30:00Z",
        end_utc="2026-03-09T16:45:00Z",
        participant_ids=["carol", "dave"],
    )).status_code == 202

    assert _cancel(client, alice_slot["reservation_id"]).status_code == 200
    assert client.get("/api/v1/reservation-waitlist/wl-other").json()["status"] == "pending"


def test_cancel_version_mismatch_does_not_promote(client):
    blocked_by = client.post("/api/v1/reservations", json=_resv_payload(
        request_id="req-block", participant_ids=["alice"])).json()
    assert client.post("/api/v1/reservation-waitlist", json=_wl_payload(
        request_id="wl-1",
        start_utc="2026-03-09T13:30:00Z",
        end_utc="2026-03-09T13:45:00Z",
        participant_ids=["alice"],
    )).status_code == 202

    resp = _cancel(client, blocked_by["reservation_id"], expected_version=5)
    assert resp.status_code == 409
    assert client.get("/api/v1/reservation-waitlist/wl-1").json()["status"] == "pending"
    # 被取消的预约仍占用时段（候补未被递补）
    blocked = client.post("/api/v1/reservations", json=_resv_payload(
        request_id="req-check",
        start_utc="2026-03-09T13:30:00Z",
        end_utc="2026-03-09T13:45:00Z",
        participant_ids=["alice"],
    ))
    assert blocked.status_code == 409


def test_cancel_promotes_after_lazily_expiring_hold(client, db_path):
    """挡住候补的保留到期后，取消另一挡路预约时同事务放行递补。"""
    # bob 的正式预约 + alice 的 1 秒保留，同时挡住 alice+bob 的候补
    bob_slot = client.post("/api/v1/reservations", json=_resv_payload(
        request_id="req-bob", participant_ids=["bob"])).json()
    hold = client.post("/api/v1/reservation-holds", json=_hold_payload(
        ttl_seconds=1)).json()
    assert client.post("/api/v1/reservation-waitlist", json=_wl_payload(
        request_id="wl-1",
        start_utc="2026-03-09T13:30:00Z",
        end_utc="2026-03-09T13:45:00Z",
        participant_ids=["alice", "bob"],
    )).status_code == 202

    _force_hold_expired(db_path, hold["hold_id"])
    assert _cancel(client, bob_slot["reservation_id"]).status_code == 200

    got = client.get("/api/v1/reservation-waitlist/wl-1").json()
    assert got["status"] == "promoted"
    assert got["reservation"]["participants"] == ["alice", "bob"]


def test_cancel_without_waitlist_works_as_before(client):
    created = client.post("/api/v1/reservations", json=_resv_payload()).json()
    resp = _cancel(client, created["reservation_id"])
    assert resp.status_code == 200
    assert resp.json()["status"] == "canceled"


def test_fifo_tie_break_by_id_when_created_at_equal(client, db_path):
    """created_at 同刻时 FIFO 以 id 决胜（确定性，不依赖返回顺序）。"""
    blocked_by = client.post("/api/v1/reservations", json=_resv_payload(
        request_id="req-block", participant_ids=["alice"])).json()
    assert client.post("/api/v1/reservation-waitlist", json=_wl_payload(
        request_id="wl-1",
        start_utc="2026-03-09T13:30:00Z",
        end_utc="2026-03-09T13:45:00Z",
        participant_ids=["alice"],
    )).status_code == 202
    assert client.post("/api/v1/reservation-waitlist", json=_wl_payload(
        request_id="wl-2",
        start_utc="2026-03-09T13:30:00Z",
        end_utc="2026-03-09T13:45:00Z",
        participant_ids=["alice"],
    )).status_code == 202
    # 把两条候补的 created_at 强制改成同一时刻
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "UPDATE reservation_waitlist SET created_at = ?"
            " WHERE request_id IN ('wl-1', 'wl-2')",
            ("2026-03-09T10:00:00.000000Z",))
        conn.commit()
    finally:
        conn.close()

    assert _cancel(client, blocked_by["reservation_id"]).status_code == 200
    assert client.get("/api/v1/reservation-waitlist/wl-1").json()["status"] == "promoted"
    assert client.get("/api/v1/reservation-waitlist/wl-2").json()["status"] == "pending"


def test_promoted_reservation_can_be_canceled_and_audited(client):
    """入队直接成约的预约可正常取消，链为 created -> canceled。"""
    promoted = client.post("/api/v1/reservation-waitlist",
                           json=_wl_payload()).json()
    rid = promoted["reservation"]["reservation_id"]
    resp = _cancel(client, rid, expected_version=1)
    assert resp.status_code == 200
    assert resp.json()["status"] == "canceled"
    history = client.get(f"/api/v1/reservations/{rid}/history").json()
    assert [e["action"] for e in history["events"]] == ["created", "canceled"]
    # 候补行状态保持 promoted（取消的是关联预约，不影响候补终态标记）
    got = client.get("/api/v1/reservation-waitlist/wl-1").json()
    assert got["status"] == "promoted"
    assert got["reservation"]["status"] == "canceled"


# ------------------------------------------------------------------ 并发

def test_concurrent_same_request_id_joins_single_entry(db_path):
    """8 个并发同内容候补（空闲时段）：全部得到同一 promoted 候补与同一预约。"""
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        return waitlist_core.join_waitlist(
            request_id="wl-same",
            start_utc=_utc(2026, 3, 9, 13),
            end_utc=_utc(2026, 3, 9, 14),
            participant_ids=["alice", "bob"],
            db_path=db_path,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(lambda _: worker(), range(8)))

    ids = {e.id for e, _ in outcomes}
    assert len(ids) == 1
    assert sum(1 for _, created in outcomes if created) == 1
    assert {e.status for e, _ in outcomes} == {"promoted"}

    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM reservation_waitlist").fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM reservations").fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM reservation_events WHERE action = 'created'"
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_concurrent_cancel_promotion_vs_direct_booking_single_occupant(db_path):
    """取消递补与并发直接抢占串行化：该时段最终至多一条有效 alice 占用。"""
    blocked, _ = reservation_core.create_reservation(
        request_id="req-block",
        start_utc=_utc(2026, 3, 9, 13),
        end_utc=_utc(2026, 3, 9, 14),
        participant_ids=["alice"],
        db_path=db_path,
    )
    waitlist_core.join_waitlist(
        request_id="wl-1",
        start_utc=_utc(2026, 3, 9, 13, 30),
        end_utc=_utc(2026, 3, 9, 13, 45),
        participant_ids=["alice"],
        db_path=db_path,
    )
    barrier = threading.Barrier(6)

    def canceler():
        barrier.wait()
        return reservation_core.cancel_reservation(
            reservation_id=blocked.id, expected_version=1, db_path=db_path)

    def booker(i: int):
        barrier.wait()
        try:
            return reservation_core.create_reservation(
                request_id=f"req-grab-{i}",
                start_utc=_utc(2026, 3, 9, 13, 30),
                end_utc=_utc(2026, 3, 9, 13, 45),
                participant_ids=["alice"],
                db_path=db_path,
            )
        except reservation_core.ReservationConflictError:
            return None

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(canceler)] + [pool.submit(booker, i) for i in range(5)]
        results = [f.result() for f in futures]

    # 取消实际执行一次（唯一的 canceler）；直接抢占有 0 或 1 个成功，
    # 与候补递补合计恰好 1 条占用
    canceled_result = results[0]
    assert canceled_result is not None and canceled_result[1] is True
    direct_wins = [r for r in results[1:] if r is not None]
    assert len(direct_wins) <= 1

    conn = sqlite3.connect(db_path)
    try:
        active = conn.execute(
            "SELECT COUNT(*) FROM reservation_participants"
            " WHERE participant_id = 'alice'"
            "   AND start_utc = '2026-03-09T13:30:00.000000Z'"
            "   AND end_utc = '2026-03-09T13:45:00.000000Z'"
        ).fetchone()[0]
        assert active == 1
        status = conn.execute(
            "SELECT status FROM reservation_waitlist WHERE request_id = 'wl-1'"
        ).fetchone()[0]
        assert status in ("pending", "promoted")
        # 候补递补与直接抢占互斥
        assert (status == "promoted") == (len(direct_wins) == 0)
    finally:
        conn.close()


# ------------------------------------------------------------------ 核心层

def test_core_get_unknown_raises(db_path):
    with pytest.raises(waitlist_core.WaitlistNotFoundError):
        waitlist_core.get_waitlist_entry("missing", db_path=db_path)


def test_core_join_queue_and_promote_roundtrip(db_path):
    reservation_core.create_reservation(
        request_id="req-block",
        start_utc=_utc(2026, 3, 9, 13),
        end_utc=_utc(2026, 3, 9, 14),
        participant_ids=["alice"],
        db_path=db_path,
    )
    entry, created = waitlist_core.join_waitlist(
        request_id="wl-core",
        start_utc=_utc(2026, 3, 9, 13, 30),
        end_utc=_utc(2026, 3, 9, 13, 45),
        participant_ids=["bob", "alice"],
        db_path=db_path,
    )
    assert created is True and entry.status == "pending"
    assert entry.participants == ("alice", "bob")

    again, created2 = waitlist_core.join_waitlist(
        request_id="wl-core",
        start_utc=_utc(2026, 3, 9, 13, 30),
        end_utc=_utc(2026, 3, 9, 13, 45),
        participant_ids=["alice", "bob"],
        db_path=db_path,
    )
    assert created2 is False and again.id == entry.id and again.status == "pending"


# -------------------------------------------------------- 旧数据库无损建新表

def test_init_creates_waitlist_tables_losslessly(tmp_path, monkeypatch):
    """旧版库（只有 reservations/holds 等表）初始化后新增候补表，既有数据不动。"""
    path = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(path)
    try:
        conn.executescript("""\
CREATE TABLE reservations (
    id INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT NOT NULL UNIQUE,
    start_utc TEXT NOT NULL, end_utc TEXT NOT NULL,
    participants_key TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE reservation_participants (
    reservation_id INTEGER NOT NULL REFERENCES reservations(id) ON DELETE CASCADE,
    participant_id TEXT NOT NULL, start_utc TEXT NOT NULL, end_utc TEXT NOT NULL,
    PRIMARY KEY (reservation_id, participant_id));
""")
        conn.execute(
            "INSERT INTO reservations"
            " (request_id, start_utc, end_utc, participants_key, created_at)"
            " VALUES ('old', '2026-03-09T13:00:00.000000Z',"
            " '2026-03-09T14:00:00.000000Z', '[\"alice\"]',"
            " '2026-03-01T00:00:00.000000Z')")
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setenv("RESERVATIONS_DB_PATH", path)
    db_module.init_db(path)
    db_module.init_db(path)  # 迁移幂等

    conn = sqlite3.connect(path)
    try:
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"reservation_waitlist",
                "reservation_waitlist_participants"} <= names
        assert conn.execute(
            "SELECT request_id, participants_key FROM reservations"
        ).fetchone() == ("old", "[\"alice\"]")
    finally:
        conn.close()

    # 新表立即可用（空闲时段直接成约）
    entry, created = waitlist_core.join_waitlist(
        request_id="wl-new",
        start_utc=_utc(2026, 5, 1, 13),
        end_utc=_utc(2026, 5, 1, 14),
        participant_ids=["alice"],
        db_path=path,
    )
    assert created is True
    assert entry.status == "promoted"
    assert entry.reservation is not None and entry.reservation.request_id == "wl-new"
