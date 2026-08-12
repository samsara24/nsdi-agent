"""M7 可执行断言校验器。

约束库里的 `formal_expression` 是给人看的。这里把其中**可机械判定**的部分写成断言，
用来校验两类对象：

1. **证据包本身**（`check_evidence`）。`invariant` 类约束断言的是器件物理上必然成立的
   关系，违反它说明数据有问题而不是故障证据。这类违规不该拿去推理，该去查采集。
2. **LLM 的每一步推理**（`check_response`）。这是 T6 的验收要求「LLM 每步输出可被约束校验」。

校验分四层，按严重程度递减：

- `fabricated_evidence`：引用了证据包里不存在的 token。这是幻觉，一票否决。
  模型可以编造措辞，但编不出一个不在证据包里的 token 而不被发现。
- `constraint_violation`：结论违反了排除类约束（例如在本端未发光时仍判 fiber）。
- `forbidden_claim`：说了 `caveat` 类约束明令禁止的话（例如给出绝对链路损耗数值）。
- `unsupported_step`：某一步既没引证据也没引约束，属于凭空断言。

`Violation.severity` 决定处置：`fatal` 必须重写或弃权，`warning` 只记录不拦截。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..anomaly import DOWN_THRESHOLDS, lane_values
from ..evidence_pack import EvidencePack
from ..types import ROOT_CAUSES, SIDES
from .library import CONSTRAINT_LIBRARY, ConstraintLibrary
from .semantics import matches_scope


SEVERITIES: Tuple[str, ...] = ("fatal", "warning")

VIOLATION_KINDS: Tuple[str, ...] = (
    "fabricated_evidence",
    "fabricated_constraint",
    "constraint_violation",
    "forbidden_claim",
    "unsupported_step",
    "invalid_measurement",
)

#: `caveat` 类约束里可以机械检测的禁止说法。
#: 键是约束 id，值是（正则, 说明）。正则刻意写得保守，宁可漏检也不误伤——
#: 误伤会让合规的推理被反复重写，代价比漏检大。
FORBIDDEN_CLAIM_PATTERNS: Dict[str, Tuple[str, str]] = {
    "C12_no_absolute_link_loss": (
        r"(链路|线路|光纤)?(损耗|衰减)\s*(约|大约|为|是|=|达到)?\s*-?\d+(\.\d+)?\s*(dB|db|分贝)",
        "给出了绝对链路损耗数值。两端功率相减在本数据集上会得到负损耗，该量不可信。",
    ),
    "C13_serdes_snr_unit_unknown": (
        r"serdes[_ ]?snr[^。；;]{0,20}?-?\d+(\.\d+)?\s*(dB|db|分贝)",
        "把 serdes_snr 当作 dB 量纲讨论。该字段量纲未知，只能作有效 / 失效二值使用。",
    ),
    "C14_host_snr_mostly_missing": (
        r"host[_ ]?snr[^。；;]{0,10}?(正常|良好|健康|无异常)",
        "把缺失的 host_snr 当作正常。缺失只能表述为「未采集」。",
    ),
    "C15_blackout_sentinel_is_not_laser_off": (
        r"(未发光|没有发光|激光关断|laser\s*off|停止发光)",
        "在全链路遥测失效时断言某端未发光。此时哨兵表示读不到数，不是没有光。",
    ),
}

#: `C19` 的结构化判据：这些词出现在一个 `effect == "support"` 的步骤里，
#: 说明模型在用群体统计支持结论。用「词 + support」两个条件同时成立才判违规，
#: 是为了不误伤「本 case 的观测与训练集分布一致」这类合法描述。
PRIOR_AS_SUPPORT_PATTERN = r"(先验|基础比例|多数类|最常见|统计上更多|训练集(中|里)?(更|占)|SOP\s*叶|叶节点)"


@dataclass(frozen=True)
class Violation:
    kind: str
    severity: str
    message: str
    constraint_id: str = ""
    step_index: Optional[int] = None
    detail: str = ""

    def __post_init__(self) -> None:
        if self.kind not in VIOLATION_KINDS:
            raise ValueError(f"unknown violation kind: {self.kind}")
        if self.severity not in SEVERITIES:
            raise ValueError(f"unknown severity: {self.severity}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "severity": self.severity,
            "message": self.message,
            "constraint_id": self.constraint_id,
            "step_index": self.step_index,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class CheckReport:
    violations: Tuple[Violation, ...] = ()

    @property
    def fatal(self) -> Tuple[Violation, ...]:
        return tuple(item for item in self.violations if item.severity == "fatal")

    @property
    def ok(self) -> bool:
        return not self.fatal

    def feedback(self) -> str:
        """给模型的重写提示。只列 fatal，且必须说清楚哪一步错了、为什么错。"""
        lines = []
        for item in self.fatal:
            where = f"第 {item.step_index + 1} 步" if item.step_index is not None else "整体结论"
            source = f"（违反 {item.constraint_id}）" if item.constraint_id else ""
            lines.append(f"- {where}{source}：{item.message}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "fatal_count": len(self.fatal),
            "violations": [item.to_dict() for item in self.violations],
        }


# --- 证据包自校验 -----------------------------------------------------------

def check_evidence(
    pack: EvidencePack,
    *,
    library: ConstraintLibrary = CONSTRAINT_LIBRARY,
) -> CheckReport:
    """校验证据包本身是否违反 invariant 类约束。

    违反 invariant 说明数据有问题，不是故障证据，因此严重程度是 `warning`：
    它不该阻断推理，但必须让推理者知道这条读数不可信。
    """
    violations: List[Violation] = []

    for side in SIDES:
        temperature = pack.scalars.get(f"{side}.Temperature")
        if temperature is not None and not (0.0 <= temperature <= 70.0):
            violations.append(Violation(
                kind="invalid_measurement", severity="warning",
                constraint_id="C3_temperature_operating_range",
                message=f"{side} 侧温度 {temperature:.2f} degC 超出 0-70 degC 工作范围",
            ))
        voltage = pack.scalars.get(f"{side}.Voltage")
        if voltage is not None and not (3.135 <= voltage <= 3.465):
            violations.append(Violation(
                kind="invalid_measurement", severity="warning",
                constraint_id="C4_voltage_nominal_band",
                message=f"{side} 侧电压 {voltage:.3f} V 超出 3.3 V ±5% 范围",
            ))

        for metric, bounds, constraint_id in (
            ("txpower", (-1.8, 2.1), "C5_tx_power_range"),
            ("rxpower", (-12.3, 3.0), "C7_rx_power_range"),
        ):
            sentinel = DOWN_THRESHOLDS[metric]
            healthy = [
                value for value in lane_values(pack.telemetry, metric, side).values()
                if value is not None and value > sentinel
            ]
            outside = [value for value in healthy if not (bounds[0] <= value <= bounds[1])]
            if outside:
                violations.append(Violation(
                    kind="invalid_measurement", severity="warning",
                    constraint_id=constraint_id,
                    message=(
                        f"{side} 侧 {metric} 有 {len(outside)} 个非断光读数落在实测区间 "
                        f"[{bounds[0]}, {bounds[1]}] 之外"
                    ),
                    detail=f"越界值={sorted(outside)}",
                ))

    if pack.optical_blackout:
        violations.append(Violation(
            kind="invalid_measurement", severity="warning",
            constraint_id="C15_blackout_sentinel_is_not_laser_off",
            message=(
                "两端收发光功率全部处于断光哨兵而 TxLOS 仍报 Normal，"
                "该状态下哨兵表示读不到数而非无光，所有断光类证据均不可用"
            ),
        ))
    return CheckReport(violations=tuple(violations))


# --- LLM 输出校验 -----------------------------------------------------------

def check_response(
    response: Any,
    pack: EvidencePack,
    available_evidence: Sequence[str],
    *,
    allowed_root_causes: Sequence[str] = ROOT_CAUSES,
    library: ConstraintLibrary = CONSTRAINT_LIBRARY,
) -> CheckReport:
    """逐步校验 LLM 输出。`response` 是 `llm.protocol.DiagnosisResponse`。"""
    violations: List[Violation] = []
    evidence = set(available_evidence)
    constraint_ids = set(library.ids())
    blackout = pack.optical_blackout

    for index, step in enumerate(response.steps):
        unknown_tokens = sorted(set(step.cited_evidence) - evidence)
        if unknown_tokens:
            violations.append(Violation(
                kind="fabricated_evidence", severity="fatal", step_index=index,
                message=(
                    f"引用了本 case 不存在的证据：{'、'.join(unknown_tokens)}。"
                    f"只能引用给定证据清单里的 token"
                ),
                detail=f"可用证据={sorted(evidence)}",
            ))
        unknown_constraints = sorted(set(step.cited_constraints) - constraint_ids)
        if unknown_constraints:
            violations.append(Violation(
                kind="fabricated_constraint", severity="fatal", step_index=index,
                message=f"引用了不存在的约束编号：{'、'.join(unknown_constraints)}",
            ))
        for constraint_id in step.cited_constraints:
            if constraint_id not in constraint_ids:
                continue
            constraint = library.get(constraint_id)
            effect = getattr(step, "effect", "neutral")
            target = getattr(step, "target", None)
            # 「这条约束的前件在本 case 上不成立」是一个合法且有信息量的观察。
            # 模型表达它的方式是 effect=neutral、target=""、并引用那条约束。
            # 这是引用方式不合规范，不是物理断言错误，因此只记 warning：
            # 迭代 3 实测有 35 步因此被判废，连带作废了整份 verdict 正确的回答。
            cites_as_negative = effect == "neutral" and not target
            if effect not in constraint.allowed_effects:
                violations.append(Violation(
                    kind="constraint_violation",
                    severity="warning" if cites_as_negative else "fatal",
                    step_index=index,
                    constraint_id=constraint_id,
                    message=(
                        f"该约束不允许 effect={effect}；"
                        f"允许值为 {', '.join(constraint.allowed_effects)}"
                        + ("。中性步骤应当不引用任何约束" if cites_as_negative else "")
                    ),
                ))
            if target not in constraint.allowed_targets and not cites_as_negative:
                violations.append(Violation(
                    kind="constraint_violation", severity="fatal", step_index=index,
                    constraint_id=constraint_id,
                    message=(
                        f"该约束不允许 target={target or ''}；"
                        f"允许值为 {', '.join(constraint.allowed_targets)}"
                    ),
                ))
            matching_tokens = [
                token for token in step.cited_evidence
                if matches_scope(token, constraint.applies_to_token_prefixes)
            ] if constraint.applies_to_token_prefixes else []
            if constraint.applies_to_token_prefixes and not matching_tokens:
                violations.append(Violation(
                    kind="constraint_violation", severity="fatal", step_index=index,
                    constraint_id=constraint_id,
                    message="引用的证据 token 家族与该约束的适用范围不匹配",
                    detail=(
                        f"expected prefixes={constraint.applies_to_token_prefixes}"
                        "（同侧同指标的其他判据族也接受，见 constraints.semantics）"
                    ),
                ))
            if not constraint.applies_to_token_prefixes and step.cited_evidence:
                violations.append(Violation(
                    kind="constraint_violation", severity="fatal", step_index=index,
                    constraint_id=constraint_id,
                    message="该约束在 v2 中没有对应 token，只能作为不绑定证据 token 的中性上下文引用",
                    detail=f"cited evidence={step.cited_evidence}",
                ))
            if constraint_id == "C6_tx_down_excludes_medium" and blackout:
                violations.append(Violation(
                    kind="constraint_violation", severity="fatal", step_index=index,
                    constraint_id=constraint_id,
                    message="全链路遥测失效时 C6 的“本端未发光”前提不成立，不能据此排除 fiber",
                ))
            if constraint_id == "C15_blackout_sentinel_is_not_laser_off" and not blackout:
                violations.append(Violation(
                    kind="constraint_violation", severity="fatal", step_index=index,
                    constraint_id=constraint_id,
                    message="当前 case 未命中全链路 blackout，不能引用 C15 解释哨兵语义",
                ))
        if not step.cited_evidence and not step.cited_constraints:
            violations.append(Violation(
                kind="unsupported_step", severity="fatal", step_index=index,
                message="这一步既没有引用证据也没有引用约束，属于凭空断言",
                detail=step.claim,
            ))
        violations.extend(_forbidden_claims(step.claim, index, blackout))
        if getattr(step, "effect", "neutral") == "support" and re.search(
            PRIOR_AS_SUPPORT_PATTERN, step.claim
        ):
            violations.append(Violation(
                kind="forbidden_claim", severity="fatal", step_index=index,
                constraint_id="C19_population_prior_is_not_case_evidence",
                message=(
                    "用群体统计（类别先验 / SOP 叶节点分布 / 历史标签投票）作为 support 步骤的依据。"
                    "群体统计只能决定默认动作，不能当作本 case 的物理证据"
                ),
                detail=step.claim,
            ))

    verdict = response.verdict
    if verdict is not None:
        if verdict not in allowed_root_causes:
            violations.append(Violation(
                kind="constraint_violation", severity="fatal",
                constraint_id="C6_tx_down_excludes_medium",
                message=(
                    f"结论 {verdict} 已被确定性物理排除，"
                    f"可选根因只有 {'、'.join(allowed_root_causes)}"
                ),
            ))
        if blackout:
            violations.append(Violation(
                kind="constraint_violation", severity="fatal",
                constraint_id="C15_blackout_sentinel_is_not_laser_off",
                message=(
                    "本 case 全链路遥测失效，观测无法区分根因，不得给出结论，应当弃权并请求现场确认"
                ),
            ))
        if not response.steps:
            violations.append(Violation(
                kind="unsupported_step", severity="fatal",
                message="给出了结论但没有任何推理步骤",
            ))

    return CheckReport(violations=tuple(violations))


def _forbidden_claims(text: str, index: int, blackout: bool) -> List[Violation]:
    found: List[Violation] = []
    for constraint_id, (pattern, message) in FORBIDDEN_CLAIM_PATTERNS.items():
        # C15 的禁止说法只在全链路失效时成立；正常 case 讨论「未发光」是合法的。
        if constraint_id == "C15_blackout_sentinel_is_not_laser_off" and not blackout:
            continue
        if re.search(pattern, text, flags=re.IGNORECASE):
            found.append(Violation(
                kind="forbidden_claim", severity="fatal", step_index=index,
                constraint_id=constraint_id, message=message, detail=text,
            ))
    return found
