"""M9 统一置信度、LLM 标定与降级决策。

历史匹配分支的置信度来自训练集留一法；LLM 分支不能直接把模型自报的 confidence
当作正确率。本模块把 LLM 输出按分支与 confidence 分桶，在独立的训练留一法输出上
统计正确率和 Wilson 下界，再用同一套出口决定最终结论、补采或人工介入。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .branches.base import BranchOutcome, wilson_lower_bound


DECISION_POLICY_VERSION = "decision-policy-v3-forced-multidim"
LLM_CONFIDENCE_BINS: Tuple[float, ...] = (0.0, 0.5, 0.7, 0.9, 1.0)
DECISION_ACTIONS: Tuple[str, ...] = ("final", "request_evidence", "human_review")

#: M9 可用的候选来源。
#:
#: `branch` 是历史匹配或 LLM 得出的 case 特异结论；`sop` 是训练集归纳路径的叶节点先验。
#: 两者的可靠性口径不同但都来自训练集：前者是 train-LOO 分组频率，
#: 后者是叶节点自身的标签分布。按照 `docs/个人整体思路.md`，正式主链路
#: 只能默认接受 `branch`；`sop` 可作为显式消融、报告字段或 N5c 的统计先验，
#: 不能替代证据图匹配与专家 SOP 约束下的 LLM 推理。
#:
#: `expert` 是现网人工经验规则（`rca_framework.expert`），迭代 3 加入。它与前两者
#: 有一处根本不同：**规则本身不从本数据学任何参数**，训练集只用来统计各规则组的
#: 可靠性。它证明归因方向知识有价值，但正式方法不能让专家规则级联顶替证据图
#: 历史匹配主干；因此只允许在对照实验中显式加入。
CANDIDATE_SOURCES: Tuple[str, ...] = ("branch", "sop", "expert")


def llm_calibration_group(branch: str, confidence: float) -> str:
    value = min(1.0, max(0.0, float(confidence)))
    for lower, upper in zip(LLM_CONFIDENCE_BINS[:-1], LLM_CONFIDENCE_BINS[1:]):
        if value < upper or upper == 1.0:
            closing = "]" if upper == 1.0 else ")"
            return f"llm:{branch}:[{lower:.1f},{upper:.1f}{closing}"
    return f"llm:{branch}:[0.9,1.0]"


@dataclass(frozen=True)
class LLMCalibration:
    """LLM 已回答样本的独立可靠性标定表。"""

    counts: Mapping[str, Tuple[int, int]] = field(default_factory=dict)
    source: str = ""

    @classmethod
    def fit(
        cls,
        outcomes: Sequence[BranchOutcome],
        traces: Sequence[Optional[Any]],
        labels: Sequence[str],
        *,
        source: str = "train-loo",
    ) -> "LLMCalibration":
        if not (len(outcomes) == len(traces) == len(labels)):
            raise ValueError("outcomes, traces and labels must be the same length")
        tally: Dict[str, list[int]] = {}
        for outcome, trace, truth in zip(outcomes, traces, labels):
            accepted = getattr(trace, "accepted", None) if trace is not None else None
            if accepted is None or accepted.verdict is None:
                continue
            group = llm_calibration_group(outcome.branch, accepted.confidence)
            row = tally.setdefault(group, [0, 0])
            row[0] += int(accepted.verdict == truth)
            row[1] += 1
        return cls(
            counts={key: (value[0], value[1]) for key, value in sorted(tally.items())},
            source=source,
        )

    def confidence(self, group: str) -> float:
        correct, total = self.counts.get(group, (0, 0))
        return round(correct / total, 6) if total else 0.0

    def lower_bound(self, group: str) -> float:
        correct, total = self.counts.get(group, (0, 0))
        return wilson_lower_bound(correct, total)

    def support(self, group: str) -> int:
        return self.counts.get(group, (0, 0))[1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "confidence_bins": list(LLM_CONFIDENCE_BINS),
            "groups": {
                key: {
                    "correct": correct,
                    "total": total,
                    "accuracy": round(correct / total, 6) if total else 0.0,
                    "wilson_lower_bound": wilson_lower_bound(correct, total),
                }
                for key, (correct, total) in sorted(self.counts.items())
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LLMCalibration":
        return cls(
            counts={
                str(key): (int(item["correct"]), int(item["total"]))
                for key, item in value.get("groups", {}).items()
            },
            source=str(value.get("source", "")),
        )


def apply_llm_calibration(
    outcome: BranchOutcome,
    trace: Optional[Any],
    calibration: Optional[LLMCalibration],
) -> BranchOutcome:
    """保留 LLM 多维原始分，同时并行记录 Wilson 标定分组。"""
    accepted = getattr(trace, "accepted", None) if trace is not None else None
    if accepted is None or accepted.verdict is None:
        return outcome
    group = llm_calibration_group(outcome.branch, accepted.confidence)
    support = calibration.support(group) if calibration is not None else 0
    if support:
        return replace(
            outcome,
            confidence_lower_bound=calibration.lower_bound(group),
            calibration_group=group,
            calibration_support=support,
        )
    caveat = "LLM 结论尚无独立标定样本；模型自报置信度仅供记录，不能直接作为最终可靠性"
    return replace(
        outcome,
        confidence=accepted.confidence,
        confidence_lower_bound=0.0,
        calibration_group=f"uncalibrated:{group}",
        calibration_support=0,
        caveats=outcome.caveats + (() if caveat in outcome.caveats else (caveat,)),
    )


@dataclass(frozen=True)
class DecisionCandidate:
    """M9 待评估的一个候选结论及其可靠性口径。"""

    source: str
    verdict: Optional[str]
    confidence: float
    confidence_lower_bound: float
    support: int
    group: str
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "verdict": self.verdict,
            "confidence": round(self.confidence, 6),
            "confidence_lower_bound": round(self.confidence_lower_bound, 6),
            "support": self.support,
            "group": self.group,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class DecisionPolicy:
    """N6 统一出口策略。

    `final_lower_bound` 的取值不应当靠拍板。v1 写死 0.5 与支持数 10，
    在 161 条训练 case 分成 7 个标定组之后，任何组都不可能让 Wilson 95% 下界
    达到 0.5——想在 p=0.63 处让下界过 0.5 需要约 50 个同组样本。
    结果是安全门禁把 100% 的 case 挡在外面，系统不产出任何结论。

    v2 把阈值改成「在训练留一法上按目标选择性风险反解出来的工作点」。
    按个人整体思路，正式默认不再退到 SOP 或 expert 候选；`fitted_on` 记录这个
    工作点是怎么定出来的；没有拟合过程时它为空，表示阈值是人工指定的。
    """

    version: str = DECISION_POLICY_VERSION
    final_lower_bound: float = 0.5
    minimum_support: int = 10
    candidate_order: Tuple[str, ...] = ("branch",)
    target_selective_risk: Optional[float] = None
    fitted_on: str = ""
    #: 在信息层面不可识别的根因。落在这里的候选永远不能成为自动结论，
    #: 而是转成带定向补采清单的 `request_evidence`。
    #: 依据是 C20：现有遥测里 fiber 的最强富集条件 Wilson 下界只有 8.2%，
    #: 与 7.45% 的先验无法区分，因此任何 fiber 结论都是在猜。
    #: 这与「把 fiber 预测删掉提高准确率」不是一回事：候选与理由都保留在报告里，
    #: 只是出口从「结论」改成「需要哪一项现场测量」。
    non_identifiable_labels: Tuple[str, ...] = ()
    #: 命中 `non_identifiable_labels` 时给出的定向补采项。
    non_identifiable_evidence: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    #: 按预测类别分别设定的下界。留空时所有类别共用 `final_lower_bound`。
    #:
    #: 单一门限在类别先验差一倍的数据上会结构性地偏向多数类：L2 先验 62.1%，
    #: 任何指向 L2 的候选起点就比指向 L1（先验 30.4%）的候选高一截，
    #: 于是一个统一门限会先把 L1 候选全部挡掉。迭代 1 的实测正是如此——
    #: 门限 0.4104 下 L1 召回只有 6.25%、平衡召回 0.2596 低于随机猜一类的 1/3，
    #: 而整体精度靠 L2 撑到 70.42%。按类别校准是把「每一类的风险都达标」
    #: 写进目标，而不是让多数类替少数类背书。
    per_label_lower_bound: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = sorted(set(self.candidate_order) - set(CANDIDATE_SOURCES))
        if unknown:
            raise ValueError(f"unknown candidate sources: {unknown}")
        if not self.candidate_order:
            raise ValueError("candidate_order must not be empty")
        for label, bound in self.per_label_lower_bound.items():
            if not 0.0 <= float(bound) <= 1.0:
                raise ValueError(f"per-label lower bound for {label} must be in [0, 1]: {bound}")

    def lower_bound_for(self, label: Optional[str]) -> float:
        if label is not None and label in self.per_label_lower_bound:
            return float(self.per_label_lower_bound[label])
        return self.final_lower_bound

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "final_lower_bound": round(self.final_lower_bound, 6),
            "minimum_support": self.minimum_support,
            "candidate_order": list(self.candidate_order),
            "target_selective_risk": self.target_selective_risk,
            "fitted_on": self.fitted_on,
            "non_identifiable_labels": list(self.non_identifiable_labels),
            "non_identifiable_evidence": {
                key: list(value) for key, value in sorted(self.non_identifiable_evidence.items())
            },
            "per_label_lower_bound": {
                key: round(float(value), 6)
                for key, value in sorted(self.per_label_lower_bound.items())
            },
        }


#: C20 认定 fiber 不可识别时需要补采的介质侧测量。这些都不在当前遥测里，
#: 因此它同时是一份「要让 fiber 变得可判别，必须新增哪些采集」的需求清单。
FIBER_EVIDENCE_REQUEST: Tuple[str, ...] = (
    "OTDR 曲线（定位反射与损耗事件的距离）",
    "两端 MPO / LC 端面镜检结果",
    "同一 lane 的双向功率标定（用于替代不可信的功率相减）",
    "光纤跳线更换后的复测结果",
)

DEFAULT_DECISION_POLICY = DecisionPolicy()


def sop_candidate(sop_prediction: Optional[Mapping[str, Any]]) -> Optional[DecisionCandidate]:
    """把训练归纳树叶节点包装成候选。

    兼容旧 learned SOP 与新 numeric decision tree。叶节点的支持数与 Wilson 下界
    都来自训练集标签分布，因此它是群体先验，不是当前 case 的物理证据。
    """
    if not sop_prediction or sop_prediction.get("verdict") is None:
        return None
    return DecisionCandidate(
        source="sop",
        verdict=str(sop_prediction["verdict"]),
        confidence=float(sop_prediction.get("confidence", 0.0)),
        confidence_lower_bound=float(sop_prediction.get("confidence_lower_bound", 0.0)),
        support=int(sop_prediction.get("support", 0)),
        group=(
            f"tree_leaf:{sop_prediction.get('leaf_id', '')}"
            if str(sop_prediction.get("model", "")).startswith("numeric-decision-tree")
            else f"sop_leaf:{sop_prediction.get('leaf_id', '')}"
        ),
        reason=str(sop_prediction.get("reason", "")),
    )


def expert_candidate(
    expert_prediction: Optional[Mapping[str, Any]],
) -> Optional[DecisionCandidate]:
    """把专家规则的裁决包装成候选。

    与 SOP 候选的关键区别在于 `group` 的含义：SOP 的组是「训练集里落在同一叶子的
    那批 case」，专家规则的组是「命中同一条规则的那批 case」。后者是**因果同类**
    而不是统计同类——同组 case 共享的是一条物理归因链，不是一段特征区间。
    这也是它的可靠性能跨 train/test 稳住的原因。
    """
    if not expert_prediction or expert_prediction.get("verdict") is None:
        return None
    return DecisionCandidate(
        source="expert",
        verdict=str(expert_prediction["verdict"]),
        confidence=float(expert_prediction.get("confidence", 0.0)),
        confidence_lower_bound=float(expert_prediction.get("confidence_lower_bound", 0.0)),
        support=int(expert_prediction.get("support", 0)),
        group=str(expert_prediction.get("group", "expert:unknown")),
        reason=str(expert_prediction.get("reason", "")),
    )


def branch_candidate(outcome: BranchOutcome) -> Optional[DecisionCandidate]:
    if outcome.verdict is None:
        return None
    return DecisionCandidate(
        source="branch",
        verdict=outcome.verdict,
        confidence=outcome.confidence,
        confidence_lower_bound=outcome.confidence_lower_bound,
        support=outcome.calibration_support,
        group=outcome.calibration_group,
        reason=f"{outcome.branch} 分支结论",
    )


def build_candidates(
    outcome: BranchOutcome,
    *,
    sop_prediction: Optional[Mapping[str, Any]] = None,
    expert_prediction: Optional[Mapping[str, Any]] = None,
    policy: DecisionPolicy = DEFAULT_DECISION_POLICY,
) -> Tuple[DecisionCandidate, ...]:
    builders = {
        "branch": lambda: branch_candidate(outcome),
        "sop": lambda: sop_candidate(sop_prediction),
        "expert": lambda: expert_candidate(expert_prediction),
    }
    candidates = []
    for source in policy.candidate_order:
        candidate = builders[source]()
        if candidate is not None:
            candidates.append(candidate)
    return tuple(candidates)


def simulate_gate(
    rows: Sequence[Tuple[Sequence[DecisionCandidate], str]],
    policy: DecisionPolicy,
) -> Dict[str, Any]:
    """在给定策略下模拟一次门禁，返回覆盖率与选择性风险。

    模拟必须走与 `decide` 完全相同的级联逻辑，否则拟合出来的阈值
    在真实推理时会得到不同结果。
    """
    answered = 0
    correct = 0
    by_source: Dict[str, list[int]] = {}
    by_label: Dict[str, list[int]] = {}
    recall_hits: Dict[str, int] = {}
    truth_counts: Dict[str, int] = {}
    for candidates, truth in rows:
        truth_counts[truth] = truth_counts.get(truth, 0) + 1
        for candidate in candidates:
            if not passes_gate(candidate, policy):
                continue
            answered += 1
            hit = int(candidate.verdict == truth)
            correct += hit
            entry = by_source.setdefault(candidate.source, [0, 0])
            entry[0] += hit
            entry[1] += 1
            label_entry = by_label.setdefault(str(candidate.verdict), [0, 0])
            label_entry[0] += hit
            label_entry[1] += 1
            if hit:
                recall_hits[truth] = recall_hits.get(truth, 0) + 1
            break
    total = len(rows)
    recalls = [
        recall_hits.get(label, 0) / count
        for label, count in truth_counts.items()
        if count
    ]
    return {
        "answered": answered,
        "coverage": round(answered / total, 6) if total else 0.0,
        "correct": correct,
        "precision_when_answered": round(correct / answered, 6) if answered else None,
        "selective_risk": round(1.0 - correct / answered, 6) if answered else None,
        # 按**预测类别**拆分的风险。按类别校准门限需要它：整体风险达标不代表
        # 每一类都达标，迭代 1 就是整体 29.6% 风险下 L1 召回只有 6.25%。
        "by_predicted_label": {
            label: {
                "answered": value[1],
                "correct": value[0],
                "precision": round(value[0] / value[1], 6) if value[1] else None,
                "selective_risk": round(1.0 - value[0] / value[1], 6) if value[1] else None,
            }
            for label, value in sorted(by_label.items())
        },
        "balanced_recall": round(sum(recalls) / len(recalls), 6) if recalls else 0.0,
        "by_source": {
            source: {"correct": value[0], "answered": value[1]}
            for source, value in sorted(by_source.items())
        },
    }


def fit_decision_policy(
    rows: Sequence[Tuple[Sequence[DecisionCandidate], str]],
    *,
    target_selective_risk: float = 0.30,
    minimum_support: int = 10,
    minimum_coverage: float = 0.0,
    candidate_order: Tuple[str, ...] = ("branch",),
    non_identifiable_labels: Tuple[str, ...] = (),
    non_identifiable_evidence: Optional[Mapping[str, Tuple[str, ...]]] = None,
    source: str = "train-loo",
    class_conditional: bool = False,
    class_conditional_rounds: int = 2,
) -> Tuple[DecisionPolicy, Dict[str, Any]]:
    """在训练留一法结果上反解出 `final_lower_bound`。

    `rows` 是每条训练 case 的 `(候选级联, 真值)`。目标写成
    「在选择性风险不超过 `target_selective_risk` 的前提下取最大覆盖率」，
    这是运维能直接理解的口径：允许多少比例的自动结论是错的。
    v1 那个抽象的 0.5 下界没有对应任何可讨论的业务约束。

    `class_conditional=True` 时，在统一门限之上再按预测类别逐类校准
    （见 `refine_per_label_bounds`），把「每一类的风险都达标」写进目标；
    否则单一门限会让多数类的正确率替少数类背书。

    返回策略与整条阈值-覆盖率曲线。找不到满足目标的阈值时返回最严格阈值，
    并在 `fitted_on` 里写明这件事，绝不悄悄放宽目标。
    """
    if not 0.0 < target_selective_risk < 1.0:
        raise ValueError("target_selective_risk must be in (0, 1)")
    blocked = tuple(non_identifiable_labels)
    evidence_map = dict(non_identifiable_evidence or {})
    thresholds = sorted(
        {0.0}
        | {
            round(candidate.confidence_lower_bound, 6)
            for candidates, _ in rows
            for candidate in candidates
            if candidate.support >= minimum_support and candidate.verdict not in blocked
        }
    )
    curve = []
    best: Optional[Tuple[float, float]] = None
    for threshold in thresholds:
        probe = DecisionPolicy(
            final_lower_bound=threshold,
            minimum_support=minimum_support,
            candidate_order=candidate_order,
            non_identifiable_labels=blocked,
            non_identifiable_evidence=evidence_map,
        )
        stats = simulate_gate(rows, probe)
        curve.append(
            {
                "final_lower_bound": threshold,
                "answered": stats["answered"],
                "coverage": stats["coverage"],
                "correct": stats["correct"],
                "precision_when_answered": stats["precision_when_answered"],
                "selective_risk": stats["selective_risk"],
                "balanced_recall": stats["balanced_recall"],
                "by_source": stats["by_source"],
            }
        )
        risk = stats["selective_risk"]
        if risk is None or stats["coverage"] < minimum_coverage:
            continue
        if risk <= target_selective_risk and (best is None or stats["coverage"] > best[1]):
            best = (threshold, stats["coverage"])

    if best is None:
        chosen = thresholds[-1] if thresholds else 0.0
        fitted_on = (
            f"{source}: 目标选择性风险 {target_selective_risk:.2%} 在训练留一法上无可行阈值，"
            f"退到最严格候选阈值 {chosen:.4f}"
        )
    else:
        chosen = best[0]
        matched = next(item for item in curve if item["final_lower_bound"] == chosen)
        fitted_on = (
            f"{source}: 目标选择性风险 <= {target_selective_risk:.2%}，"
            f"取到最大覆盖率的阈值 {chosen:.4f}"
            f"（训练留一法覆盖率 {matched['coverage']:.2%}，"
            f"实测风险 {matched['selective_risk']:.2%}，支持数下限 {minimum_support}）"
        )
    policy = DecisionPolicy(
        final_lower_bound=chosen,
        minimum_support=minimum_support,
        candidate_order=candidate_order,
        target_selective_risk=target_selective_risk,
        fitted_on=fitted_on,
        non_identifiable_labels=blocked,
        non_identifiable_evidence=evidence_map,
    )
    diagnostics: Dict[str, Any] = {
        "source": source,
        "target_selective_risk": target_selective_risk,
        "minimum_support": minimum_support,
        "minimum_coverage": minimum_coverage,
        "candidate_order": list(candidate_order),
        "non_identifiable_labels": list(blocked),
        "chosen_lower_bound": chosen,
        "feasible": best is not None,
        "curve": curve,
    }
    if class_conditional:
        policy, refinement = refine_per_label_bounds(
            rows,
            policy,
            target_selective_risk=target_selective_risk,
            rounds=class_conditional_rounds,
            source=source,
        )
        diagnostics["class_conditional"] = refinement
    return policy, diagnostics


def refine_per_label_bounds(
    rows: Sequence[Tuple[Sequence[DecisionCandidate], str]],
    policy: DecisionPolicy,
    *,
    target_selective_risk: float,
    rounds: int = 2,
    source: str = "train-loo",
) -> Tuple[DecisionPolicy, Dict[str, Any]]:
    """在统一门限之上，为每个预测类别单独收紧或放宽下界。

    做法是坐标上升：固定其它类别的门限，对当前类别扫遍所有候选下界，
    取「该类风险 <= 目标」里覆盖最大的一个；轮换若干轮直到稳定。
    之所以不做联合最优，是因为候选级联会在第一个过门的候选处 break，
    改一个类别的门限会改变别的类别看到的样本，联合搜索既慢又更容易过拟合；
    坐标上升的每一步都能解释成「这一类当前的风险是多少、为此把门限挪到哪」。

    关键约束：**只有该类自己的风险达标才放宽它**。这样放宽 L1 门限
    不会靠 L2 的正确率来掩盖 L1 的错误，也就避免了迭代 1 那种
    「整体风险达标但少数类几乎没有召回」的结果。
    """
    labels = sorted(
        {
            str(candidate.verdict)
            for candidates, _ in rows
            for candidate in candidates
            if candidate.verdict is not None
            and candidate.verdict not in policy.non_identifiable_labels
        }
    )
    thresholds_by_label: Dict[str, list[float]] = {
        label: sorted(
            {0.0}
            | {
                round(candidate.confidence_lower_bound, 6)
                for candidates, _ in rows
                for candidate in candidates
                if str(candidate.verdict) == label
                and candidate.support >= policy.minimum_support
            }
        )
        for label in labels
    }

    bounds: Dict[str, float] = {label: policy.final_lower_bound for label in labels}
    history: list[Dict[str, Any]] = []
    for round_index in range(max(1, rounds)):
        changed = False
        for label in labels:
            best_choice: Optional[Tuple[float, int]] = None
            for threshold in thresholds_by_label[label]:
                probe_bounds = dict(bounds)
                probe_bounds[label] = threshold
                stats = simulate_gate(rows, replace(policy, per_label_lower_bound=probe_bounds))
                row = stats["by_predicted_label"].get(label)
                if not row or not row["answered"]:
                    continue
                if row["selective_risk"] is None or row["selective_risk"] > target_selective_risk:
                    continue
                if best_choice is None or row["answered"] > best_choice[1]:
                    best_choice = (threshold, row["answered"])
            if best_choice is not None and best_choice[0] != bounds[label]:
                bounds[label] = best_choice[0]
                changed = True
        stats = simulate_gate(rows, replace(policy, per_label_lower_bound=dict(bounds)))
        history.append(
            {
                "round": round_index,
                "bounds": {label: round(value, 6) for label, value in sorted(bounds.items())},
                "coverage": stats["coverage"],
                "selective_risk": stats["selective_risk"],
                "balanced_recall": stats["balanced_recall"],
                "by_predicted_label": stats["by_predicted_label"],
            }
        )
        if not changed:
            break

    final_stats = simulate_gate(rows, replace(policy, per_label_lower_bound=dict(bounds)))
    unmet = sorted(
        label
        for label, row in final_stats["by_predicted_label"].items()
        if row["selective_risk"] is not None and row["selective_risk"] > target_selective_risk
    )
    refined = replace(
        policy,
        per_label_lower_bound=dict(bounds),
        fitted_on=(
            f"{policy.fitted_on}；按类别校准后下界 "
            + "、".join(f"{label}={bounds[label]:.4f}" for label in labels)
            + f"（训练留一法覆盖率 {final_stats['coverage']:.2%}，"
            f"整体风险 {(final_stats['selective_risk'] or 0):.2%}，"
            f"平衡召回 {final_stats['balanced_recall']:.4f}）"
        ),
    )
    return refined, {
        "source": source,
        "target_selective_risk": target_selective_risk,
        "rounds_run": len(history),
        "bounds": {label: round(value, 6) for label, value in sorted(bounds.items())},
        "labels_missing_target": unmet,
        "history": history,
        "final": {
            "coverage": final_stats["coverage"],
            "selective_risk": final_stats["selective_risk"],
            "balanced_recall": final_stats["balanced_recall"],
            "by_predicted_label": final_stats["by_predicted_label"],
        },
    }


@dataclass(frozen=True)
class FinalDecision:
    case_id: str
    branch: str
    action: str
    verdict: Optional[str]
    proposed_verdict: Optional[str]
    confidence: float
    confidence_lower_bound: float
    calibration_group: str
    calibration_support: int
    reason: str
    requested_evidence: Tuple[str, ...] = ()
    candidate_source: str = "branch"
    considered_candidates: Tuple[DecisionCandidate, ...] = ()
    confidence_breakdown: Optional[Mapping[str, float]] = None
    history_verdict: Optional[str] = None
    fallback_source: str = ""
    compliance_penalties: Tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "branch": self.branch,
            "action": self.action,
            "verdict": self.verdict,
            "proposed_verdict": self.proposed_verdict,
            "confidence": self.confidence,
            "confidence_lower_bound": self.confidence_lower_bound,
            "calibration_group": self.calibration_group,
            "calibration_support": self.calibration_support,
            "reason": self.reason,
            "requested_evidence": list(self.requested_evidence),
            "candidate_source": self.candidate_source,
            "considered_candidates": [item.to_dict() for item in self.considered_candidates],
            "confidence_breakdown": dict(self.confidence_breakdown or {}),
            "history_verdict": self.history_verdict,
            "fallback_source": self.fallback_source,
            "compliance_penalties": [dict(item) for item in self.compliance_penalties],
        }


#: 报告里对每个候选来源的说明。区分它们不是措辞问题：三者的证据强度不同，
#: 运维看到「群体先验」和看到「本 case 的物理归因链」应当采取不同的复核动作。
CANDIDATE_ORIGINS: Mapping[str, str] = {
    "branch": "分支证据链",
    "sop": "learned SOP 叶节点先验（群体统计，不是本 case 的物理证据）",
    "expert": "专家规则归因链（现网人工经验，方向由物理链路结构决定）",
}


def passes_gate(candidate: DecisionCandidate, policy: DecisionPolicy) -> bool:
    if candidate.verdict is None:
        return False
    if candidate.group.startswith(("llm:", "llm_raw:", "uncalibrated:")):
        return candidate.confidence >= policy.lower_bound_for(candidate.verdict)
    return (
        candidate.support >= policy.minimum_support
        and candidate.confidence_lower_bound >= policy.lower_bound_for(candidate.verdict)
    )


def decide(
    outcome: BranchOutcome,
    policy: DecisionPolicy = DEFAULT_DECISION_POLICY,
    *,
    sop_prediction: Optional[Mapping[str, Any]] = None,
    expert_prediction: Optional[Mapping[str, Any]] = None,
) -> FinalDecision:
    """把任意 N5 输出收敛成最终结论、补采请求或人工介入。

    候选按 `policy.candidate_order` 依次过门禁，第一个通过的胜出。
    正式默认只接受 `branch`，也就是历史匹配分支或 LLM 仲裁后的 case 特异证据链。
    `sop` / `expert` 只能通过显式 `candidate_order` 加入对照或消融。
    """
    candidates = build_candidates(
        outcome,
        sop_prediction=sop_prediction,
        expert_prediction=expert_prediction,
        policy=policy,
    )
    for candidate in candidates:
        if not passes_gate(candidate, policy):
            continue
        origin = CANDIDATE_ORIGINS.get(candidate.source, candidate.source)
        return FinalDecision(
            case_id=outcome.case_id,
            branch=outcome.branch,
            action="final",
            verdict=candidate.verdict,
            proposed_verdict=candidate.verdict,
            confidence=candidate.confidence,
            confidence_lower_bound=candidate.confidence_lower_bound,
            calibration_group=candidate.group,
            calibration_support=candidate.support,
            reason=(
                f"采用{origin}：多维综合置信度 {candidate.confidence:.2%} 达到"
                f"阈值 {policy.lower_bound_for(candidate.verdict):.2%}"
            ),
            candidate_source=candidate.source,
            considered_candidates=candidates,
            confidence_breakdown=outcome.confidence_breakdown,
            history_verdict=outcome.history_verdict,
            fallback_source=outcome.fallback_source,
            compliance_penalties=outcome.compliance_penalties,
        )

    best = candidates[0] if candidates else None
    evidence_score = float((outcome.confidence_breakdown or {}).get("evidence_completeness", 1.0))
    if evidence_score < 0.3 or outcome.missing_evidence:
        action = "request_evidence"
        reason = "候选未达到自动结案阈值，且证据完整度偏低；先补齐缺失证据再重新诊断"
    else:
        action = "human_review"
        if best is None:
            reason = "当前路径未形成可校验结论，且没有明确的自动补采项，转人工介入"
        else:
            reason = (
                f"最优候选（来源 {best.source}）未通过可靠性门槛"
                f"（多维综合置信度 {best.confidence:.2%}），转人工复核"
            )
    return FinalDecision(
        case_id=outcome.case_id,
        branch=outcome.branch,
        action=action,
        verdict=None,
        proposed_verdict=best.verdict if best is not None else None,
        confidence=best.confidence if best is not None else outcome.confidence,
        confidence_lower_bound=(
            best.confidence_lower_bound if best is not None else outcome.confidence_lower_bound
        ),
        calibration_group=best.group if best is not None else outcome.calibration_group,
        calibration_support=best.support if best is not None else outcome.calibration_support,
        reason=reason,
        requested_evidence=outcome.missing_evidence if action == "request_evidence" else (),
        candidate_source=best.source if best is not None else "none",
        considered_candidates=candidates,
        confidence_breakdown=outcome.confidence_breakdown,
        history_verdict=outcome.history_verdict,
        fallback_source=outcome.fallback_source,
        compliance_penalties=outcome.compliance_penalties,
    )


def decide_many(
    outcomes: Sequence[BranchOutcome],
    policy: DecisionPolicy = DEFAULT_DECISION_POLICY,
    *,
    sop_predictions: Optional[Sequence[Optional[Mapping[str, Any]]]] = None,
    expert_predictions: Optional[Sequence[Optional[Mapping[str, Any]]]] = None,
) -> Tuple[FinalDecision, ...]:
    if sop_predictions is not None and len(sop_predictions) != len(outcomes):
        raise ValueError("sop_predictions must be the same length as outcomes")
    if expert_predictions is not None and len(expert_predictions) != len(outcomes):
        raise ValueError("expert_predictions must be the same length as outcomes")
    return tuple(
        decide(
            outcome,
            policy,
            sop_prediction=None if sop_predictions is None else sop_predictions[index],
            expert_prediction=None if expert_predictions is None else expert_predictions[index],
        )
        for index, outcome in enumerate(outcomes)
    )

