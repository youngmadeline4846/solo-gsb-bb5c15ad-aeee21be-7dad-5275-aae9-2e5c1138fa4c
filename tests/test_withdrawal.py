"""封签事件撤回影响预演接口的单元测试与 API 集成测试。"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.models import Event, EventType, WithdrawalPreviewRequest
from app.withdrawal import preview_withdrawal

client = TestClient(app)

DAY = "2026-09-19"


def dt(hour: int, minute: int = 0) -> datetime:
    base = datetime.fromisoformat(f"{DAY}T00:00:00+00:00")
    return base + timedelta(hours=hour, minutes=minute)


def ev(type_: EventType, hour: int, minute: int = 0, **kw) -> Event:
    return Event(type=type_, timestamp=dt(hour, minute), **kw)


def make_request(events, candidates) -> WithdrawalPreviewRequest:
    return WithdrawalPreviewRequest(events=list(events), candidates=list(candidates))


def scenario_by_removal(resp, removed_index):
    """按被移除下标取方案（与候选输入顺序无关的断言辅助）。"""
    for scenario in resp.scenarios:
        if scenario.removed_index == removed_index:
            return scenario
    raise AssertionError(f"缺少移除 {removed_index} 的方案")


# ---------------------------------------------------------------------------
# 基线与逐方案校验
# ---------------------------------------------------------------------------


def test_baseline_preserved_and_scenarios_in_candidate_order():
    # 0:seal ok / 1:unseal ok / 2:reseal 复用 S1 violation
    resp = preview_withdrawal(
        make_request(
            events=[
                ev(EventType.SEAL, 10, seal_id="S1", custodian="Alice"),
                ev(EventType.UNSEAL, 11),
                ev(EventType.RESEAL, 12, seal_id="S1", custodian="Alice"),
            ],
            candidates=[2, 0],
        )
    )
    assert resp.baseline.valid is False
    assert resp.baseline.sealed is False
    assert resp.baseline.final_custodian == "Alice"
    assert resp.baseline.current_seal is None
    assert [e.status for e in resp.baseline.events] == ["ok", "ok", "violation"]
    assert [v.index for v in resp.baseline.violations] == [2]
    # 方案按候选输入顺序返回
    assert [s.removed_index for s in resp.scenarios] == [2, 0]


def test_removing_violation_event_yields_clean_chain():
    resp = preview_withdrawal(
        make_request(
            events=[
                ev(EventType.SEAL, 10, seal_id="S1", custodian="Alice"),
                ev(EventType.UNSEAL, 11),
                ev(EventType.RESEAL, 12, seal_id="S1", custodian="Alice"),
            ],
            candidates=[2],
        )
    )
    scenario = resp.scenarios[0]
    assert scenario.removed_index == 2
    assert scenario.removed_event.type is EventType.RESEAL
    assert scenario.removed_event.seal_id == "S1"
    assert scenario.validation.valid is True
    assert scenario.validation.sealed is False
    assert scenario.validation.final_custodian == "Alice"
    assert scenario.ok_count == 2
    assert scenario.violation_count == 0
    assert scenario.resolved == []
    assert scenario.introduced == []
    assert resp.recommended == 0


def test_indices_mapped_back_to_original_chain():
    # 移除下标 1 后，缩短链下标 0/1/2 应映射回原链 0/2/3
    resp = preview_withdrawal(
        make_request(
            events=[
                ev(EventType.SEAL, 10, seal_id="S1", custodian="Alice"),
                ev(EventType.HANDOVER, 11, custodian="Bob"),
                ev(EventType.HANDOVER, 10, 30, custodian="Carol"),  # 时间不增，violation
                ev(EventType.UNSEAL, 12),
            ],
            candidates=[1],
        )
    )
    scenario = resp.scenarios[0]
    assert [e.index for e in scenario.validation.events] == [0, 2, 3]
    # 基线下标 2 的交接因前驱 11 点交接被移除而变为有效：resolved=[2]
    assert scenario.resolved == [2]
    assert scenario.introduced == []
    assert scenario.ok_count == 3
    assert scenario.violation_count == 0
    assert scenario.validation.valid is True
    assert scenario.validation.sealed is False
    assert scenario.validation.final_custodian == "Carol"


def test_violation_indices_mapped_back_to_original_chain():
    # 移除下标 0 后，缩短链位置 1 的违规应映射回原链下标 2
    resp = preview_withdrawal(
        make_request(
            events=[
                ev(EventType.UNSEAL, 10),  # 未封装开封，violation
                ev(EventType.SEAL, 11, seal_id="S1", custodian="Alice"),
                ev(EventType.SEAL, 12, seal_id="S2", custodian="Bob"),  # 已封装，violation
            ],
            candidates=[0],
        )
    )
    scenario = resp.scenarios[0]
    assert [e.index for e in scenario.validation.events] == [1, 2]
    assert [e.status for e in scenario.validation.events] == ["ok", "violation"]
    assert [v.index for v in scenario.validation.violations] == [2]
    assert "已处于封装状态" in scenario.validation.violations[0].reason
    assert scenario.ok_count == 1
    assert scenario.violation_count == 1
    # 下标 2 在基线中也是 violation：既不 resolved 也不 introduced
    assert scenario.resolved == []
    assert scenario.introduced == []


def test_removed_event_not_mixed_into_diffs():
    # 基线唯一下标 0 违规；移除后链变干净，但 0 是被移除事件，不得计入 resolved
    resp = preview_withdrawal(
        make_request(
            events=[
                ev(EventType.HANDOVER, 10, custodian="Bob"),  # 未封装交接，violation
                ev(EventType.SEAL, 11, seal_id="S1", custodian="Alice"),
            ],
            candidates=[0, 1],
        )
    )
    removed0 = scenario_by_removal(resp, 0)
    assert removed0.validation.valid is True
    assert [e.index for e in removed0.validation.events] == [1]
    assert removed0.resolved == []
    assert removed0.introduced == []
    assert removed0.ok_count == 1

    removed1 = scenario_by_removal(resp, 1)
    assert [e.index for e in removed1.validation.events] == [0]
    assert [e.status for e in removed1.validation.events] == ["violation"]
    # 下标 0 在基线中同样是 violation：不 introduced
    assert removed1.resolved == []
    assert removed1.introduced == []
    assert removed1.ok_count == 0
    assert removed1.violation_count == 1


def test_introduced_violations_listed():
    # 移除 unseal 后，后续 reseal 因仍处于封装状态而违规
    resp = preview_withdrawal(
        make_request(
            events=[
                ev(EventType.SEAL, 10, seal_id="S1", custodian="Alice"),
                ev(EventType.UNSEAL, 11),
                ev(EventType.RESEAL, 12, seal_id="S2", custodian="Alice"),
            ],
            candidates=[1],
        )
    )
    scenario = resp.scenarios[0]
    assert [e.index for e in scenario.validation.events] == [0, 2]
    assert [e.status for e in scenario.validation.events] == ["ok", "violation"]
    assert scenario.introduced == [2]
    assert scenario.resolved == []
    assert scenario.ok_count == 1
    assert scenario.violation_count == 1
    # 移除后最终状态：仍封装在 S1 下
    assert scenario.validation.sealed is True
    assert scenario.validation.current_seal == "S1"


def test_removing_only_event_yields_empty_chain():
    resp = preview_withdrawal(
        make_request(
            events=[ev(EventType.SEAL, 10, seal_id="S1", custodian="Alice")],
            candidates=[0],
        )
    )
    scenario = resp.scenarios[0]
    assert scenario.validation.valid is True
    assert scenario.validation.sealed is False
    assert scenario.validation.final_custodian is None
    assert scenario.validation.current_seal is None
    assert scenario.validation.events == []
    assert scenario.validation.violations == []
    assert scenario.ok_count == 0
    assert scenario.violation_count == 0
    assert scenario.resolved == []
    assert scenario.introduced == []
    assert resp.recommended == 0


# ---------------------------------------------------------------------------
# 推荐方案：有效事件数 → introduced 数 → 最终违规数 → 候选输入顺序
# ---------------------------------------------------------------------------


def test_recommendation_maximizes_ok_count():
    # 复用封签链：移除违规的 reseal（下标 2）得到有效事件最多的方案
    resp = preview_withdrawal(
        make_request(
            events=[
                ev(EventType.SEAL, 10, seal_id="S1", custodian="Alice"),
                ev(EventType.UNSEAL, 11),
                ev(EventType.RESEAL, 12, seal_id="S1", custodian="Alice"),
            ],
            candidates=[0, 1, 2],
        )
    )
    counts = {s.removed_index: (s.ok_count, s.violation_count) for s in resp.scenarios}
    assert counts == {0: (0, 2), 1: (1, 1), 2: (2, 0)}
    assert resp.recommended == 2
    assert resp.scenarios[resp.recommended].removed_index == 2


def test_recommendation_minimizes_introduced_on_ok_count_tie():
    # 两个方案有效事件数与最终违规数相同（ok=4 / viol=1）：
    # 移除 2 会级联出下标 4 的新违规（introduced=1），移除 1 无任何差异（introduced=0）
    events = [
        ev(EventType.SEAL, 10, seal_id="S1", custodian="Alice"),
        ev(EventType.HANDOVER, 10, 30, custodian="Bob"),
        ev(EventType.HANDOVER, 11, custodian="Carol"),
        ev(EventType.UNSEAL, 10, 45),  # 时间不增（10:45 <= 11:00），violation
        ev(EventType.UNSEAL, 12),
        ev(EventType.RESEAL, 13, seal_id="S2", custodian="Dave"),
    ]
    resp = preview_withdrawal(make_request(events=events, candidates=[2, 1]))
    removed2, removed1 = resp.scenarios
    # 移除 2：下标 3 时间戳恢复有效（resolved=[3]），但其开封使下标 4 变为
    # 未封装开封（introduced=[4]）
    assert (removed2.ok_count, removed2.violation_count) == (4, 1)
    assert removed2.resolved == [3]
    assert removed2.introduced == [4]
    # 移除 1：不影响任何其他事件的判定
    assert (removed1.ok_count, removed1.violation_count) == (4, 1)
    assert removed1.resolved == []
    assert removed1.introduced == []
    # 有效事件数相同，introduced 更少者胜（即使输入顺序靠后）
    assert resp.recommended == 1
    assert resp.scenarios[resp.recommended].removed_index == 1

    # 交换候选输入顺序，推荐仍指向移除下标 1 的方案
    resp_swapped = preview_withdrawal(make_request(events=events, candidates=[1, 2]))
    assert resp_swapped.recommended == 0
    assert resp_swapped.scenarios[0].removed_index == 1


def test_recommendation_full_tie_falls_back_to_candidate_input_order():
    # 移除 1 或 2 完全并列（ok=2 / introduced=0 / viol=1）：按候选输入顺序取首项
    events = [
        ev(EventType.SEAL, 10, seal_id="S1", custodian="Alice"),
        ev(EventType.HANDOVER, 9, custodian="Bob"),  # 时间不增，violation
        ev(EventType.HANDOVER, 9, 30, custodian="Carol"),  # 时间不增，violation
        ev(EventType.UNSEAL, 14),
    ]
    resp = preview_withdrawal(make_request(events=events, candidates=[2, 1]))
    for scenario in resp.scenarios:
        assert (scenario.ok_count, scenario.violation_count) == (2, 1)
        assert scenario.resolved == []
        assert scenario.introduced == []
    assert resp.recommended == 0
    assert resp.scenarios[0].removed_index == 2

    resp_swapped = preview_withdrawal(make_request(events=events, candidates=[1, 2]))
    assert resp_swapped.recommended == 0
    assert resp_swapped.scenarios[0].removed_index == 1


def test_recommendation_resolved_events_do_not_count_for_removed():
    # 基线违规恰为被移除事件时不计入 resolved，两方案并列按输入顺序
    events = [
        ev(EventType.SEAL, 10, seal_id="S1", custodian="Alice"),
        ev(EventType.SEAL, 11, seal_id="S2", custodian="Bob"),  # 已封装，violation
        ev(EventType.UNSEAL, 12),
        ev(EventType.SEAL, 13, seal_id="S3", custodian="Carol"),
    ]
    resp = preview_withdrawal(make_request(events=events, candidates=[0, 1]))
    removed0, removed1 = resp.scenarios
    # 移除 0：下标 1 的 seal 由 violation 变 ok（resolved=[1]）
    assert removed0.resolved == [1]
    assert removed0.ok_count == 3
    # 移除 1：基线违规事件被移除，不计入 resolved
    assert removed1.resolved == []
    assert removed1.ok_count == 3
    # ok 数、introduced 数、最终违规数均并列 → 输入顺序首项
    assert resp.recommended == 0


# ---------------------------------------------------------------------------
# 422：候选下标重复 / 越界 / 数量约束
# ---------------------------------------------------------------------------


def test_duplicate_candidates_rejected():
    with pytest.raises(ValidationError, match="候选下标不可重复"):
        make_request(
            events=[ev(EventType.SEAL, 10, seal_id="S1", custodian="Alice")],
            candidates=[0, 0],
        )


def test_out_of_range_candidate_rejected():
    with pytest.raises(ValidationError, match="候选下标越界"):
        make_request(
            events=[ev(EventType.SEAL, 10, seal_id="S1", custodian="Alice")],
            candidates=[1],
        )


def test_negative_candidate_rejected():
    with pytest.raises(ValidationError, match="候选下标越界"):
        make_request(
            events=[ev(EventType.SEAL, 10, seal_id="S1", custodian="Alice")],
            candidates=[-1],
        )


def test_candidate_on_empty_chain_rejected():
    with pytest.raises(ValidationError, match="候选下标越界"):
        make_request(events=[], candidates=[0])


def test_empty_candidates_rejected():
    with pytest.raises(ValidationError):
        make_request(
            events=[ev(EventType.SEAL, 10, seal_id="S1", custodian="Alice")],
            candidates=[],
        )


def test_more_than_20_candidates_rejected():
    events = _long_valid_chain(21)
    with pytest.raises(ValidationError):
        make_request(events=events, candidates=list(range(21)))


def test_exactly_20_candidates_accepted():
    events = _long_valid_chain(21)
    resp = preview_withdrawal(make_request(events=events, candidates=list(range(20))))
    assert len(resp.scenarios) == 20
    assert resp.baseline.valid is True
    # 移除任一 unseal 都会让后续 reseal 违规；移除末尾 reseal 无影响
    assert 0 <= resp.recommended < 20


def _long_valid_chain(count: int):
    """构造 count 个事件的全有效链：seal / (unseal, reseal) 循环。"""
    events = [ev(EventType.SEAL, 0, seal_id="S0", custodian="Alice")]
    seal_no = 0
    while len(events) < count:
        hour = len(events)
        events.append(ev(EventType.UNSEAL, hour))
        if len(events) < count:
            seal_no += 1
            events.append(
                ev(EventType.RESEAL, hour + 1, seal_id=f"S{seal_no}", custodian="Alice")
            )
    return events[:count]


# ---------------------------------------------------------------------------
# HTTP 集成测试
# ---------------------------------------------------------------------------


def _http_payload():
    return {
        "events": [
            {"type": "seal", "timestamp": f"{DAY}T10:00:00Z",
             "seal_id": "S1", "custodian": "Alice"},
            {"type": "unseal", "timestamp": f"{DAY}T11:00:00Z"},
            {"type": "reseal", "timestamp": f"{DAY}T12:00:00Z",
             "seal_id": "S1", "custodian": "Alice"},
        ],
        "candidates": [0, 2],
    }


def test_http_endpoint_success():
    response = client.post("/api/v1/seal-chain/withdrawal-preview", json=_http_payload())
    assert response.status_code == 200
    data = response.json()
    assert data["baseline"]["valid"] is False
    assert [v["index"] for v in data["baseline"]["violations"]] == [2]
    assert [s["removed_index"] for s in data["scenarios"]] == [0, 2]

    removed0 = data["scenarios"][0]
    assert removed0["removed_event"]["type"] == "seal"
    assert removed0["removed_event"]["seal_id"] == "S1"
    assert [e["index"] for e in removed0["validation"]["events"]] == [1, 2]
    assert removed0["introduced"] == [1]
    assert removed0["resolved"] == []
    assert removed0["ok_count"] == 0
    assert removed0["violation_count"] == 2

    removed2 = data["scenarios"][1]
    assert removed2["validation"]["valid"] is True
    assert [e["index"] for e in removed2["validation"]["events"]] == [0, 1]
    assert removed2["resolved"] == []
    assert removed2["introduced"] == []
    assert removed2["ok_count"] == 2
    assert removed2["violation_count"] == 0

    # 移除下标 2 有效事件数最多
    assert data["recommended"] == 1


def test_http_duplicate_candidates_returns_422():
    payload = _http_payload()
    payload["candidates"] = [0, 2, 0]
    response = client.post("/api/v1/seal-chain/withdrawal-preview", json=payload)
    assert response.status_code == 422
    assert "候选下标不可重复" in response.text


def test_http_out_of_range_candidate_returns_422():
    payload = _http_payload()
    payload["candidates"] = [3]
    response = client.post("/api/v1/seal-chain/withdrawal-preview", json=payload)
    assert response.status_code == 422
    assert "候选下标越界" in response.text


def test_http_negative_candidate_returns_422():
    payload = _http_payload()
    payload["candidates"] = [-1]
    response = client.post("/api/v1/seal-chain/withdrawal-preview", json=payload)
    assert response.status_code == 422
    assert "候选下标越界" in response.text


def test_http_empty_candidates_returns_422():
    payload = _http_payload()
    payload["candidates"] = []
    response = client.post("/api/v1/seal-chain/withdrawal-preview", json=payload)
    assert response.status_code == 422


def test_http_more_than_20_candidates_returns_422():
    payload = _http_payload()
    payload["candidates"] = list(range(21))
    response = client.post("/api/v1/seal-chain/withdrawal-preview", json=payload)
    assert response.status_code == 422


def test_http_non_integer_candidate_returns_422():
    payload = _http_payload()
    payload["candidates"] = ["x"]
    response = client.post("/api/v1/seal-chain/withdrawal-preview", json=payload)
    assert response.status_code == 422
