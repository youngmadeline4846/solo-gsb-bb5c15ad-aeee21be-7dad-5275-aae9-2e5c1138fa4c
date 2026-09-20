"""封签事件撤回影响预演接口的单元测试与 API 集成测试。"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import (
    Event,
    EventType,
    WithdrawalDiffItem,
    WithdrawalPreviewRequest,
)
from app.withdrawal import WithdrawalPreviewError, preview_withdrawal

client = TestClient(app)

T = "2026-09-18T%02d:00:00Z"


def ev(kind: EventType, hour: int, **kw) -> Event:
    return Event(type=kind, timestamp=T % hour, **kw)


def seal(hour: int, seal_id: str, custodian: str) -> Event:
    return ev(EventType.SEAL, hour, seal_id=seal_id, custodian=custodian)


def reseal(hour: int, seal_id: str, custodian: str) -> Event:
    return ev(EventType.RESEAL, hour, seal_id=seal_id, custodian=custodian)


def handover(hour: int, custodian: str) -> Event:
    return ev(EventType.HANDOVER, hour, custodian=custodian)


def unseal(hour: int) -> Event:
    return ev(EventType.UNSEAL, hour)


def preview(events: list[Event], candidates: list[int]) -> dict:
    resp = preview_withdrawal(
        WithdrawalPreviewRequest(events=events, candidate_indices=candidates)
    )
    return resp.model_dump(mode="json")


def statuses(result: dict) -> list[str]:
    return [e["status"] for e in result["events"]]


# ---------------------------------------------------------------------------
# 基础：基线、下标映射、移除事件单列
# ---------------------------------------------------------------------------


def test_baseline_matches_single_chain_validation():
    events = [
        seal(10, "S1", "Alice"),
        handover(11, "Bob"),
        unseal(12),
        reseal(13, "S2", "Bob"),
    ]
    result = preview(events, [0])
    base = result["baseline"]

    assert result["event_count"] == 4
    assert base["valid"] is True
    assert statuses(base) == ["ok"] * 4
    assert base["sealed"] is True
    assert base["final_custodian"] == "Bob"
    assert base["current_seal"] == "S2"


def test_removed_event_is_listed_separately_and_excluded_from_validation():
    events = [
        seal(10, "S1", "Alice"),
        handover(11, "Bob"),
        unseal(12),
        reseal(13, "S2", "Bob"),
    ]
    result = preview(events, [2])
    plan = result["plans"][0]

    assert plan["candidate_index"] == 2
    assert plan["removed"] == {
        "index": 2,
        "type": "unseal",
        "timestamp": T % 12,
        "seal_id": None,
        "custodian": None,
    }
    # 剩余链结果不含被移除事件，下标映射回原链：0,1,3
    assert [e["index"] for e in plan["validation"]["events"]] == [0, 1, 3]
    assert all(item["index"] != 2 for item in plan["resolved"])
    assert all(item["index"] != 2 for item in plan["introduced"])


def test_indices_remap_to_original_chain_for_all_candidate_positions():
    events = [
        seal(10, "S1", "Alice"),
        handover(11, "Bob"),
        unseal(12),
        reseal(13, "S2", "Bob"),
    ]
    for candidate, expected in [
        (0, [1, 2, 3]),
        (1, [0, 2, 3]),
        (2, [0, 1, 3]),
        (3, [0, 1, 2]),
    ]:
        result = preview(events, [candidate])
        assert [e["index"] for e in result["plans"][0]["validation"]["events"]] == expected


def test_plans_follow_candidate_input_order():
    events = [
        seal(10, "S1", "Alice"),
        handover(11, "Bob"),
        unseal(12),
    ]
    result = preview(events, [2, 0, 1])
    assert [p["candidate_index"] for p in result["plans"]] == [2, 0, 1]


# ---------------------------------------------------------------------------
# resolved / introduced 差异
# ---------------------------------------------------------------------------


def test_removing_violating_event_only_drops_it_without_other_diffs():
    # 事件 2 在未封装状态下交接（违规）；移除它后其余事件全部 ok
    events = [
        seal(10, "S1", "Alice"),
        unseal(11),
        handover(12, "Bob"),  # 未封装交接：基线 violation
        reseal(13, "S2", "Bob"),
    ]
    result = preview(events, [2])
    base = result["baseline"]
    assert statuses(base) == ["ok", "ok", "violation", "ok"]

    plan = result["plans"][0]
    # 违规事件随移除消失，不属于 resolved（resolved 只统计状态翻转的剩余事件）
    assert plan["resolved"] == []
    assert plan["introduced"] == []
    assert statuses(plan["validation"]) == ["ok", "ok", "ok"]
    assert [e["index"] for e in plan["validation"]["events"]] == [0, 1, 3]
    assert plan["final_state"] == {
        "valid": True,
        "sealed": True,
        "final_custodian": "Bob",
        "current_seal": "S2",
    }
    assert plan["effective_event_count"] == 3
    assert plan["violation_count"] == 0


def test_removing_valid_unseal_resolves_cascading_reseal_violation():
    # 去掉 unseal(12) 后，reseal(13) 变为"已封装下重复封装"违规，
    # 但原基线中 reseal(13) 是违规的（复用封签 S1）——构造 resolved 场景：
    events = [
        seal(10, "S1", "Alice"),
        unseal(11),
        reseal(12, "S1", "Alice"),  # 复用 S1：基线 violation（封签已使用）
    ]
    base = preview(events, [2])["baseline"]
    assert statuses(base) == ["ok", "ok", "violation"]

    # 移除 unseal(1)：reseal 变成"已封装状态下重复封装"——仍违规，不算 resolved
    result = preview(events, [1])
    plan = result["plans"][0]
    assert statuses(plan["validation"]) == ["ok", "violation"]
    assert plan["resolved"] == []
    assert [item["index"] for item in plan["introduced"]] == []


def test_resolved_violation_when_removed_event_was_blocking():
    # seal(10) 之后又 seal(12) 违规；中间移除 unseal 让 reseal 路径成立：
    # 构造：seal, unseal, seal(新编号 ok 实际允许)... 改为经典场景：
    # 基线 reseal 因"未曾封装不能重封"违规——不可能（seal 已存在）。
    # 用封签复用链：移除更早的 seal 使后续 reseal 成为首次 seal 语义不可行（类型固定）。
    #
    # 经典 resolved 场景：
    # seal(S1) -> seal(S2 重复封装违规)；移除第一个 seal 后，第二个 seal 成为首个事件 → ok
    events = [
        seal(10, "S1", "Alice"),
        seal(11, "S2", "Bob"),  # 基线：已封装重复封装
    ]
    base = preview(events, [1])["baseline"]
    assert statuses(base) == ["ok", "violation"]

    result = preview(events, [0])
    plan = result["plans"][0]
    assert [e["index"] for e in plan["validation"]["events"]] == [1]
    assert statuses(plan["validation"]) == ["ok"]
    assert [item["index"] for item in plan["resolved"]] == [1]
    assert plan["resolved"][0]["reason"].startswith("当前已处于封装状态")
    assert plan["introduced"] == []
    assert plan["final_state"]["final_custodian"] == "Bob"
    assert plan["final_state"]["current_seal"] == "S2"
    assert plan["effective_event_count"] == 1
    assert plan["violation_count"] == 0


def test_introduced_violation_when_removing_unseal():
    # seal -> unseal(ok) -> reseal(S2, ok)；移除 unseal 后 reseal 变为重复封装
    events = [
        seal(10, "S1", "Alice"),
        unseal(11),
        reseal(12, "S2", "Alice"),
    ]
    result = preview(events, [1])
    plan = result["plans"][0]
    assert statuses(plan["validation"]) == ["ok", "violation"]
    assert [e["index"] for e in plan["validation"]["events"]] == [0, 2]
    assert plan["resolved"] == []
    assert [item["index"] for item in plan["introduced"]] == [2]
    assert plan["introduced"][0]["reason"].startswith("当前已处于封装状态")
    assert plan["effective_event_count"] == 1
    assert plan["violation_count"] == 1


def test_timestamp_violation_resolves_when_earlier_event_removed():
    # 违规事件时间戳不递增是相对"最近有效事件"，移除更早的有效事件可解除
    events = [
        handover(12, "Bob"),   # 0: 未封装交接（基线违规，也不推进时间）
        seal(10, "S1", "Alice"),
        seal(9, "S2", "Bob"),  # 2: 时间倒序 + 重复封装（基线违规）
    ]
    base = preview(events, [0])["baseline"]
    assert statuses(base) == ["violation", "ok", "violation"]

    result = preview(events, [1])
    plan = result["plans"][0]
    # 剩余：handover(12) 仍未封装违规；seal(9) 成为首个封装事件，时间无前置 → ok
    assert [e["index"] for e in plan["validation"]["events"]] == [0, 2]
    assert statuses(plan["validation"]) == ["violation", "ok"]
    assert [item["index"] for item in plan["resolved"]] == [2]
    assert plan["introduced"] == []


def test_no_diffs_when_removing_neutral_trailing_event():
    events = [
        seal(10, "S1", "Alice"),
        handover(11, "Bob"),
        unseal(12),
        reseal(13, "S2", "Bob"),
    ]
    result = preview(events, [3])
    plan = result["plans"][0]
    assert plan["resolved"] == []
    assert plan["introduced"] == []
    assert statuses(plan["validation"]) == ["ok", "ok", "ok"]
    assert plan["final_state"]["sealed"] is False
    assert plan["final_state"]["current_seal"] is None
    assert plan["final_state"]["final_custodian"] == "Bob"


def test_removed_violation_reasons_carry_original_baseline_reason():
    events = [
        seal(10, "S1", "Alice"),
        seal(11, "S2", "Bob"),
    ]
    result = preview(events, [0])
    resolved = result["plans"][0]["resolved"][0]
    assert resolved["type"] == "seal"
    assert "封装" in resolved["reason"]


# ---------------------------------------------------------------------------
# 推荐排序
# ---------------------------------------------------------------------------


def test_recommendation_maximizes_effective_event_count():
    # 移除违规事件（事件 2）后 3 个剩余事件全 ok；移除有效 unseal 会引入新违规
    events = [
        seal(10, "S1", "Alice"),
        unseal(11),
        handover(12, "Bob"),  # 未封装交接：基线违规
        reseal(13, "S2", "Bob"),
    ]
    # 先移除有效事件 1 作为对照
    control = preview(events, [1])["plans"][0]
    assert control["effective_event_count"] == 2  # seal + reseal 有效，handover 仍违规
    assert [i["index"] for i in control["introduced"]] == [3]

    result = preview(events, [1, 2])
    plans = {p["candidate_index"]: p for p in result["plans"]}
    assert result["recommended_candidate_index"] == 2
    assert plans[2]["recommended"] is True
    assert plans[1]["recommended"] is False
    assert plans[2]["effective_event_count"] == 3


def test_input_order_decides_when_all_keys_tie():
    # 两个完全中性的移除方案（链尾连续两个未封装交接违规，移除任一个都不改变
    # 其余事件状态）：排序键完全相同 → 按候选输入顺序取首项。
    events = [
        seal(10, "S1", "Alice"),
        unseal(11),                       # ok：开封后未封装
        handover(12, "Bob"),              # 未封装交接：违规
        handover(13, "Carol"),            # 同样未封装交接：违规
    ]
    p_remove_2 = preview(events, [2])["plans"][0]
    p_remove_3 = preview(events, [3])["plans"][0]
    for plan in (p_remove_2, p_remove_3):
        assert plan["effective_event_count"] == 2
        assert plan["introduced"] == []
        assert plan["resolved"] == []
        assert plan["violation_count"] == 1

    result = preview(events, [3, 2])
    assert result["recommended_candidate_index"] == 3
    assert [p["recommended"] for p in result["plans"]] == [True, False]

    result_reversed = preview(events, [2, 3])
    assert result_reversed["recommended_candidate_index"] == 2
    assert [p["recommended"] for p in result_reversed["plans"]] == [True, False]


def test_removing_first_seal_resolves_duplicate_without_introducing_unseal_violation():
    # seal(S1,ok) - seal(S2,重复封装违规) - unseal(ok)：移除首个 seal 后，S2 的
    # seal 成为首个封装（resolved），其后 unseal 仍合法——不产生 introduced。
    events = [
        seal(10, "S1", "Alice"),
        seal(11, "S2", "Bob"),  # 重复封装：基线违规
        unseal(12),             # ok
    ]
    assert statuses(preview(events, [2])["baseline"]) == ["ok", "violation", "ok"]

    plan = preview(events, [0])["plans"][0]
    assert statuses(plan["validation"]) == ["ok", "ok"]
    assert [e["index"] for e in plan["validation"]["events"]] == [1, 2]
    assert plan["effective_event_count"] == 2
    assert [i["index"] for i in plan["resolved"]] == [1]
    assert plan["introduced"] == []
    # 移除后的最终状态：S2 封装后又被开封 → 未封装，无当前封签，保管人 Bob
    assert plan["final_state"] == {
        "valid": True,
        "sealed": False,
        "final_custodian": "Bob",
        "current_seal": None,
    }


# ---------------------------------------------------------------------------
# 推荐排序：直接针对排序键选择器的单测（第三键与逐级并列）
# ---------------------------------------------------------------------------


def _synthetic_plan(candidate: int, effective: int, introduced: int, violations: int):
    from app.models import (
        EventResult,
        RemovedEvent,
        ValidateResponse,
        Violation,
        WithdrawalFinalState,
        WithdrawalPlan,
    )
    return WithdrawalPlan(
        candidate_index=candidate,
        removed=RemovedEvent(
            index=candidate,
            type=EventType.SEAL,
            timestamp=T % 10,
            seal_id="SX",
            custodian="X",
        ),
        final_state=WithdrawalFinalState(
            valid=violations == 0, sealed=True, final_custodian="X", current_seal="SX"
        ),
        effective_event_count=effective,
        violation_count=violations,
        validation=ValidateResponse(
            valid=violations == 0,
            sealed=True,
            final_custodian="X",
            current_seal="SX",
            events=[
                EventResult(index=k, type=EventType.SEAL, status="ok")
                for k in range(effective)
            ],
            violations=[
                Violation(index=k, reason="r") for k in range(violations)
            ],
        ),
        introduced=[
            WithdrawalDiffItem.model_construct(index=k, type=EventType.SEAL, reason="r")
            for k in range(introduced)
        ],
        recommended=False,
    )


def test_selector_ranks_by_effective_then_introduced_then_violations():
    from app.withdrawal import _select_recommended

    # 有效事件数优先于一切
    plans = [
        _synthetic_plan(0, effective=3, introduced=5, violations=9),
        _synthetic_plan(1, effective=4, introduced=0, violations=0),
    ]
    assert _select_recommended(plans) == 1

    # effective 相同：introduced 少者胜（即使违规数更多）
    plans = [
        _synthetic_plan(0, effective=3, introduced=0, violations=8),
        _synthetic_plan(1, effective=3, introduced=1, violations=0),
    ]
    assert _select_recommended(plans) == 0

    # effective、introduced 相同：最终违规少者胜
    plans = [
        _synthetic_plan(0, effective=3, introduced=1, violations=2),
        _synthetic_plan(1, effective=3, introduced=1, violations=1),
    ]
    assert _select_recommended(plans) == 1

    # 三级全部并列：候选输入顺序首项
    plans = [
        _synthetic_plan(7, effective=3, introduced=1, violations=2),
        _synthetic_plan(9, effective=3, introduced=1, violations=2),
    ]
    assert _select_recommended(plans) == 0



def test_single_candidate_is_always_recommended():
    events = [seal(10, "S1", "Alice"), unseal(11)]
    result = preview(events, [1])
    assert result["recommended_candidate_index"] == 1
    assert [p["recommended"] for p in result["plans"]] == [True]


# ---------------------------------------------------------------------------
# 入参校验：422
# ---------------------------------------------------------------------------


def test_duplicate_candidate_raises():
    events = [seal(10, "S1", "Alice"), unseal(11)]
    with pytest.raises(WithdrawalPreviewError, match="重复"):
        preview(events, [0, 0])


def test_out_of_range_candidate_raises():
    events = [seal(10, "S1", "Alice")]
    with pytest.raises(WithdrawalPreviewError, match="越界"):
        preview(events, [1])
    with pytest.raises(WithdrawalPreviewError, match="越界"):
        preview(events, [-1])


def test_empty_chain_any_index_out_of_range():
    with pytest.raises(WithdrawalPreviewError, match="越界"):
        preview([], [0])


def test_more_than_twenty_candidates_rejected_at_request_boundary():
    # 数量上限 20 由请求模型约束，在请求解析阶段即返回 422（FastAPI 校验错误）
    events = [seal(hour, f"S{hour}", "Alice") for hour in range(21)]
    response = client.post(
        "/api/v1/seal-chain/withdrawal-preview",
        json=_payload(events, list(range(21))),
    )
    assert response.status_code == 422


def test_twenty_candidates_accepted():
    events = [seal(hour, f"S{hour:02d}", "Alice") for hour in range(20)]
    result = preview(events, list(range(20)))
    assert len(result["plans"]) == 20
    assert sum(1 for p in result["plans"] if p["recommended"]) == 1


# ---------------------------------------------------------------------------
# HTTP 集成
# ---------------------------------------------------------------------------


def _payload(events: list[Event], candidates: list[int]) -> dict:
    return {
        "events": [e.model_dump(mode="json") for e in events],
        "candidate_indices": candidates,
    }


def test_http_preview_happy_path():
    payload = _payload(
        [
            seal(10, "S1", "Alice"),
            unseal(11),
            reseal(12, "S2", "Alice"),
        ],
        [1],
    )
    response = client.post("/api/v1/seal-chain/withdrawal-preview", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["event_count"] == 3
    assert body["recommended_candidate_index"] == 1
    plan = body["plans"][0]
    assert plan["removed"]["index"] == 1
    assert [e["index"] for e in plan["validation"]["events"]] == [0, 2]
    assert [i["index"] for i in plan["introduced"]] == [2]


def test_http_duplicate_index_returns_422():
    payload = _payload([seal(10, "S1", "Alice")], [0, 0])
    response = client.post("/api/v1/seal-chain/withdrawal-preview", json=payload)
    assert response.status_code == 422
    assert "重复" in response.json()["detail"]


def test_http_out_of_range_returns_422():
    payload = _payload([seal(10, "S1", "Alice")], [5])
    response = client.post("/api/v1/seal-chain/withdrawal-preview", json=payload)
    assert response.status_code == 422
    assert "越界" in response.json()["detail"]


def test_http_empty_candidates_returns_422():
    payload = _payload([seal(10, "S1", "Alice")], [])
    response = client.post("/api/v1/seal-chain/withdrawal-preview", json=payload)
    assert response.status_code == 422
