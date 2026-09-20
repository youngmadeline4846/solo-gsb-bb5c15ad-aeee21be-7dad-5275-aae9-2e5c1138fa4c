"""封签事件撤回影响预演。

处理流程：
1. 复用单链校验器（``validate_events``）计算原链基线并原样保留在 ``baseline``；
2. 对每个候选下标（1～20 个，互异且不越界，否则 422）分别只移除该事件，
   对缩短后的事件链再次复用同一套单链校验器；
3. 缩短链的校验下标映射回原链：被移除事件不占位，响应中所有事件下标
   （``validation.events`` / ``validation.violations`` / ``resolved`` /
   ``introduced``）均为原链下标；被移除事件在 ``removed_index`` /
   ``removed_event`` 中单列，不混入差异；
4. 逐方案对比基线：由 violation 变为 ok 的列入 ``resolved``，由 ok 变为
   violation 的列入 ``introduced``（均不含被移除事件），并统计移除后的
   有效事件数（``ok_count``）与最终违规数（``violation_count``）；
5. 推荐方案依次最大化有效事件数、最小化 introduced 数、最小化最终违规数，
   仍并列时按候选输入顺序取首项（``recommended`` 为 ``scenarios`` 下标）。
"""
from __future__ import annotations

from typing import List, Tuple

from .models import (
    EventResult,
    ValidateResponse,
    Violation,
    WithdrawalPreviewRequest,
    WithdrawalPreviewResponse,
    WithdrawalScenario,
)
from .validator import validate_events


def _remap_validation(
    validation: ValidateResponse, index_map: List[int]
) -> ValidateResponse:
    """把缩短链校验结果中的事件下标映射回原链下标。

    ``index_map[p]`` 为缩短链位置 ``p`` 对应的原链下标；最终状态字段
    （``valid`` / ``sealed`` / ``final_custodian`` / ``current_seal``）与
    下标无关，原样保留。
    """
    events = [
        EventResult(
            index=index_map[result.index],
            type=result.type,
            status=result.status,
            reason=result.reason,
            cross_chain_conflict=result.cross_chain_conflict,
        )
        for result in validation.events
    ]
    violations = [
        Violation(index=index_map[violation.index], reason=violation.reason)
        for violation in validation.violations
    ]
    return ValidateResponse(
        valid=validation.valid,
        sealed=validation.sealed,
        final_custodian=validation.final_custodian,
        current_seal=validation.current_seal,
        events=events,
        violations=violations,
    )


def _scenario_key(scenario: WithdrawalScenario) -> Tuple[int, int, int]:
    """推荐排序键：有效事件数降序、introduced 数升序、最终违规数升序。"""
    return (-scenario.ok_count, len(scenario.introduced), scenario.violation_count)


def preview_withdrawal(request: WithdrawalPreviewRequest) -> WithdrawalPreviewResponse:
    """先计算基线，再逐候选只移除一个事件复用单链校验并对比差异。"""
    # 阶段 1：原链基线
    baseline = validate_events(request.events)
    baseline_status = {result.index: result.status for result in baseline.events}

    # 阶段 2：逐候选只移除一个事件，复用单链校验并把下标映射回原链
    scenarios: List[WithdrawalScenario] = []
    for removed_index in request.candidates:
        remaining = [
            event for i, event in enumerate(request.events) if i != removed_index
        ]
        index_map = [i for i in range(len(request.events)) if i != removed_index]
        validation = _remap_validation(validate_events(remaining), index_map)

        # 阶段 3：对比基线（被移除事件不参与差异）；validation.events 下标
        # 升序，resolved / introduced 自然保持升序
        resolved: List[int] = []
        introduced: List[int] = []
        ok_count = 0
        for result in validation.events:
            if result.status == "ok":
                ok_count += 1
                if baseline_status[result.index] == "violation":
                    resolved.append(result.index)
            elif baseline_status[result.index] == "ok":
                introduced.append(result.index)

        scenarios.append(
            WithdrawalScenario(
                removed_index=removed_index,
                removed_event=request.events[removed_index],
                validation=validation,
                resolved=resolved,
                introduced=introduced,
                ok_count=ok_count,
                violation_count=len(validation.violations),
            )
        )

    # 阶段 4：推荐方案——按排序键取最小；候选输入顺序遍历且严格小于才替换，
    # 完全并列时自然保留输入顺序首项
    recommended = 0
    best_key = _scenario_key(scenarios[0])
    for position, scenario in enumerate(scenarios[1:], start=1):
        key = _scenario_key(scenario)
        if key < best_key:
            best_key = key
            recommended = position

    return WithdrawalPreviewResponse(
        baseline=baseline, scenarios=scenarios, recommended=recommended
    )
