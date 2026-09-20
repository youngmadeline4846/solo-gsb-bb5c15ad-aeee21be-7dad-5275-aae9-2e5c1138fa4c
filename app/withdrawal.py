"""封签事件撤回影响预演。

处理流程：
1. **基线**：不对原链做任何修改，直接调用现有单链校验器
   （``validate_events``），得到基线逐事件状态、违规与最终状态；
2. **逐方案预演**：对每个候选下标，构造仅移除该一个事件的剩余链，再次复用
   同一单链校验器（状态机规则完全一致，不另立判定口径）；
3. **下标映射**：剩余链的逐事件位置通过"跳过被移除下标"一一映射回原链，
   响应中所有事件下标均为原链下标，被移除事件单列在 ``removed`` 中；
4. **差异计算**：对映射后的同名原链下标比较基线与移除后的状态——
   violation → ok 计入 ``resolved``，ok → violation 计入 ``introduced``，
   状态未翻转者不报告，被移除事件绝不混入任一差异；
5. **推荐**：依次最大化有效事件数、最小化 introduced 数、最小化最终违规数，
   仍并列时按候选输入顺序取首项。

候选下标必须为 1～20 个互异且落在 ``events`` 下标范围内的值；
重复或越界抛出 :class:`WithdrawalPreviewError`（路由层转为 HTTP 422）。
"""
from __future__ import annotations

from typing import Dict, List

from .models import (
    Event,
    EventResult,
    RemovedEvent,
    Violation,
    WithdrawalDiffItem,
    WithdrawalFinalState,
    WithdrawalPlan,
    WithdrawalPreviewRequest,
    WithdrawalPreviewResponse,
)
from .validator import validate_events


class WithdrawalPreviewError(ValueError):
    """预演入参的语义错误（在路由层转换为 HTTP 422）。"""


def _validate_candidate_indices(events: List[Event], candidates: List[int]) -> None:
    """校验候选下标数量、互异性与范围。"""
    if not 1 <= len(candidates) <= 20:
        raise WithdrawalPreviewError("candidate_indices 必须包含 1～20 个候选下标")

    seen: set[int] = set()
    for index in candidates:
        if index < 0 or index >= len(events):
            raise WithdrawalPreviewError(
                f"候选下标越界：{index}，有效范围为 0～{len(events) - 1}"
            )
        if index in seen:
            raise WithdrawalPreviewError(f"候选下标重复：{index}")
        seen.add(index)


def _original_index(new_index: int, removed_index: int) -> int:
    """移除 ``removed_index`` 后，剩余链位置 ``new_index`` 对应的原链下标。"""
    return new_index if new_index < removed_index else new_index + 1


def _remap_validation(validation, removed_index: int):
    """把移除后校验结果中的逐事件 / 违规下标全部映射回原链。"""
    events: List[EventResult] = [
        result.model_copy(update={"index": _original_index(result.index, removed_index)})
        for result in validation.events
    ]
    violations: List[Violation] = [
        violation.model_copy(
            update={"index": _original_index(violation.index, removed_index)}
        )
        for violation in validation.violations
    ]
    return validation.model_copy(update={"events": events, "violations": violations})


def _ranking_key(plan: WithdrawalPlan) -> tuple[int, int, int]:
    """推荐排序键（越小越优）：负有效事件数、introduced 数、最终违规数。

    三个目标依次为：最大化有效事件数、最小化 introduced 数、最小化最终违规数；
    第四位"候选输入顺序"由 :func:`min` 对并列项保留首次出现位置天然实现。
    """
    return (-plan.effective_event_count, len(plan.introduced), plan.violation_count)


def _select_recommended(plans: List[WithdrawalPlan]) -> int:
    """返回推荐方案在 plans（候选输入顺序）中的位置；全并列时取首项。"""
    return min(range(len(plans)), key=lambda position: _ranking_key(plans[position]))


def preview_withdrawal(request: WithdrawalPreviewRequest) -> WithdrawalPreviewResponse:
    """计算基线并逐候选方案预演单事件撤回的影响。"""
    events = request.events
    candidates = request.candidate_indices
    _validate_candidate_indices(events, candidates)

    baseline = validate_events(events)
    baseline_status: Dict[int, str] = {
        result.index: result.status for result in baseline.events
    }
    baseline_reason: Dict[int, str] = {
        violation.index: violation.reason for violation in baseline.violations
    }

    plans: List[WithdrawalPlan] = []

    for removed_index in candidates:
        remaining = [event for index, event in enumerate(events) if index != removed_index]
        validation = _remap_validation(validate_events(remaining), removed_index)

        removed_event = events[removed_index]
        resolved: List[WithdrawalDiffItem] = []
        introduced: List[WithdrawalDiffItem] = []

        for result in validation.events:
            original_index = result.index  # 已映射回原链
            before = baseline_status.get(original_index)
            if before == result.status:
                continue
            if before == "violation" and result.status == "ok":
                resolved.append(
                    WithdrawalDiffItem(
                        index=original_index,
                        type=result.type,
                        reason=baseline_reason[original_index],
                    )
                )
            elif before == "ok" and result.status == "violation":
                introduced.append(
                    WithdrawalDiffItem(
                        index=original_index,
                        type=result.type,
                        reason=result.reason,  # type: ignore[arg-type]
                    )
                )

        effective_count = sum(1 for result in validation.events if result.status == "ok")
        plans.append(
            WithdrawalPlan(
                candidate_index=removed_index,
                removed=RemovedEvent(
                    index=removed_index,
                    type=removed_event.type,
                    timestamp=removed_event.timestamp,
                    seal_id=removed_event.seal_id,
                    custodian=removed_event.custodian,
                ),
                final_state=WithdrawalFinalState(
                    valid=validation.valid,
                    sealed=validation.sealed,
                    final_custodian=validation.final_custodian,
                    current_seal=validation.current_seal,
                ),
                effective_event_count=effective_count,
                violation_count=len(validation.violations),
                validation=validation,
                resolved=resolved,
                introduced=introduced,
                recommended=False,
            )
        )

    # 推荐：最大化有效事件数 → 最小化 introduced → 最小化最终违规数；
    # 全部并列时 _select_recommended 取候选输入顺序首项。
    recommended_position = _select_recommended(plans)
    plans[recommended_position].recommended = True

    return WithdrawalPreviewResponse(
        event_count=len(events),
        baseline=baseline,
        plans=plans,
        recommended_candidate_index=plans[recommended_position].candidate_index,
    )
