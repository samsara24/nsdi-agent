"""T5 端到端评估：跑通 N1 -> N2 -> N3 -> N4 -> N5a/b/c -> N6，并对比两套路由规则。

这个脚本回答三个问题：

1. 两套路由规则各自把 case 分到哪里，各分支实际判对多少。
2. 置信度标定是否可信：按标定分组算出来的置信度，在留出测试集上兑现了没有。
3. 经过 M9 可靠性门禁与降级之后，剩下那部分的准确率是多少——
   这是「不硬猜」的代价与收益，也是 T7 降级策略要用的曲线。

标定一律只用训练集留一法；留出测试集只用于核对。
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca_framework.anomaly import fit_thresholds  # noqa: E402
from rca_framework.branches import fit_calibration, handle_many  # noqa: E402
from rca_framework.branches.base import majority_label  # noqa: E402
from rca_framework.constraints.library import CONSTRAINT_LIBRARY  # noqa: E402
from rca_framework.constraints.measurement import MEASUREMENT_CONTRACT_LIBRARY  # noqa: E402
from rca_framework.constraints.physics import PHYSICS_LIBRARY  # noqa: E402
from rca_framework.data import cases_by_manifest_split, load_cases, load_split_manifest  # noqa: E402
from rca_framework.decision import (  # noqa: E402
    CANDIDATE_SOURCES,
    DEFAULT_DECISION_POLICY,
    FIBER_EVIDENCE_REQUEST,
    DecisionPolicy,
    LLMCalibration,
    build_candidates,
    decide_many,
    fit_decision_policy,
)
from rca_framework.decision_tree.features import numeric_features_from_pack  # noqa: E402
from rca_framework.evidence_graph import (  # noqa: E402
    BOARD_POLICY,
    COVERAGE_POLICY,
    EvidenceGraph,
    match_many,
    routing_summary,
)
from rca_framework.evidence_pack import build_packs, labels_of  # noqa: E402
from rca_framework.expert import ExpertCalibration, diagnose_many  # noqa: E402
from rca_framework.filtered_rule_expert import assess_filtered_rule_expert  # noqa: E402
from rca_framework.feedback import build_case_diagnosis  # noqa: E402
from rca_framework.knowledge import (  # noqa: E402
    _out_of_fold_sop_predictions,
    out_of_fold_expert_predictions,
)
from rca_framework.report import build_report  # noqa: E402
from rca_framework.sop import (  # noqa: E402
    EXPERT_SOP_VERSION,
    expert_sop_hash,
    LEARNED_SOP_VERSION,
    learn_sop,
)
from rca_framework.features.dictionary import dictionary_for  # noqa: E402
from rca_framework.features.extractor import extract_features, fit_feature_model  # noqa: E402
from rca_framework.llm import (  # noqa: E402
    PROMPT_TEMPLATE_VERSION,
    SOP_VERSION,
    prompt_template_hash,
)
from rca_framework.types import ROOT_CAUSES  # noqa: E402
from rca_framework.topology import SOURCE_TOPOLOGIES  # noqa: E402


def branch_report(outcomes, decisions, actual: Sequence[str]) -> Dict[str, Any]:
    rows: Dict[str, Dict[str, Any]] = {}
    for outcome, decision, truth in zip(outcomes, decisions, actual):
        row = rows.setdefault(
            decision.branch,
            {"n": 0, "answered": 0, "correct": 0, "abstained": 0, "needs_llm": 0, "needs_human": 0},
        )
        row["n"] += 1
        row["needs_llm"] += int(outcome.needs_llm)
        row["needs_human"] += int(outcome.needs_human)
        if outcome.verdict is None:
            row["abstained"] += 1
        else:
            row["answered"] += 1
            row["correct"] += int(outcome.verdict == truth)
    for row in rows.values():
        row["precision_when_answered"] = (
            round(row["correct"] / row["answered"], 6) if row["answered"] else None
        )
    return rows


def calibration_check(outcomes, actual: Sequence[str]) -> Dict[str, Any]:
    """标定兑现检查：声称的置信度 vs 留出集上的实际正确率。"""
    tally: Dict[str, Dict[str, Any]] = {}
    for outcome, truth in zip(outcomes, actual):
        row = tally.setdefault(
            outcome.calibration_group,
            {"claimed_confidence": outcome.confidence, "claimed_lower_bound": outcome.confidence_lower_bound,
             "answered": 0, "correct": 0},
        )
        if outcome.verdict is not None:
            row["answered"] += 1
            row["correct"] += int(outcome.verdict == truth)
    for row in tally.values():
        row["observed_accuracy"] = (
            round(row["correct"] / row["answered"], 6) if row["answered"] else None
        )
        if row["observed_accuracy"] is not None:
            row["gap"] = round(row["observed_accuracy"] - row["claimed_confidence"], 6)
    return dict(sorted(tally.items()))


def class_metrics(predictions: Sequence[Optional[str]], actual: Sequence[str]) -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    for label in ROOT_CAUSES:
        tp = sum(prediction == label and truth == label for prediction, truth in zip(predictions, actual))
        fp = sum(prediction == label and truth != label for prediction, truth in zip(predictions, actual))
        fn = sum(prediction != label and truth == label for prediction, truth in zip(predictions, actual))
        support = sum(truth == label for truth in actual)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows[label] = {
            "support": support,
            "predicted": tp + fp,
            "true_positive": tp,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
    return rows


def _bucket(value: float, *, step: float = 0.1) -> str:
    value = max(0.0, min(1.0, float(value)))
    lower = int(value / step) * step
    if value >= 1.0:
        lower = 1.0 - step
    upper = min(1.0, lower + step)
    return f"[{lower:.1f},{upper:.1f}{']' if upper >= 1.0 else ')'}"


def confidence_reliability(outcomes, actual: Sequence[str]) -> List[Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for outcome, truth in zip(outcomes, actual):
        bucket = _bucket(outcome.confidence)
        row = rows.setdefault(bucket, {
            "bucket": bucket,
            "n": 0,
            "correct": 0,
            "mean_confidence": 0.0,
            "truth_distribution": Counter(),
            "prediction_distribution": Counter(),
        })
        row["n"] += 1
        row["correct"] += int(outcome.verdict == truth)
        row["mean_confidence"] += float(outcome.confidence)
        row["truth_distribution"][truth] += 1
        row["prediction_distribution"][outcome.verdict] += 1
    result = []
    for bucket in sorted(rows):
        row = rows[bucket]
        n = row["n"]
        result.append({
            "bucket": bucket,
            "n": n,
            "correct": row["correct"],
            "accuracy": round(row["correct"] / n, 6) if n else None,
            "mean_confidence": round(row["mean_confidence"] / n, 6) if n else 0.0,
            "truth_distribution": dict(row["truth_distribution"]),
            "prediction_distribution": {str(k): v for k, v in row["prediction_distribution"].items()},
        })
    return result


def dimension_reliability(outcomes, actual: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    dimensions = ("evidence_completeness", "physical_compliance", "reasoning_completeness", "history_similarity")
    result: Dict[str, List[Dict[str, Any]]] = {}
    for dimension in dimensions:
        rows: Dict[str, Dict[str, Any]] = {}
        for outcome, truth in zip(outcomes, actual):
            value = float((outcome.confidence_breakdown or {}).get(dimension, 0.0))
            bucket = _bucket(value)
            row = rows.setdefault(bucket, {"bucket": bucket, "n": 0, "correct": 0, "mean_score": 0.0})
            row["n"] += 1
            row["correct"] += int(outcome.verdict == truth)
            row["mean_score"] += value
        result[dimension] = [
            {
                "bucket": bucket,
                "n": row["n"],
                "correct": row["correct"],
                "accuracy": round(row["correct"] / row["n"], 6) if row["n"] else None,
                "mean_score": round(row["mean_score"] / row["n"], 6) if row["n"] else 0.0,
            }
            for bucket, row in sorted(rows.items())
        ]
    return result


def threshold_sweep(outcomes, actual: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index in range(21):
        threshold = round(index * 0.05, 2)
        selected = [
            (outcome, truth)
            for outcome, truth in zip(outcomes, actual)
            if outcome.verdict is not None and outcome.confidence >= threshold
        ]
        correct = sum(outcome.verdict == truth for outcome, truth in selected)
        predictions = [outcome.verdict if outcome.confidence >= threshold else None for outcome in outcomes]
        rows.append({
            "threshold": threshold,
            "answered": len(selected),
            "degraded": len(outcomes) - len(selected),
            "coverage": round(len(selected) / len(outcomes), 6) if outcomes else 0.0,
            "correct": correct,
            "precision_when_answered": round(correct / len(selected), 6) if selected else None,
            "class_metrics": class_metrics(predictions, actual),
        })
    return rows


def branch_class_matrix(outcomes, decisions, actual: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for outcome, decision, truth in zip(outcomes, decisions, actual):
        branch = decision.branch
        label_rows = rows.setdefault(branch, {})
        row = label_rows.setdefault(truth, {"n": 0, "answered": 0, "correct": 0})
        row["n"] += 1
        if outcome.verdict is not None:
            row["answered"] += 1
            row["correct"] += int(outcome.verdict == truth)
    for label_rows in rows.values():
        for row in label_rows.values():
            row["precision_when_answered"] = (
                round(row["correct"] / row["answered"], 6) if row["answered"] else None
            )
            row["full_recall"] = round(row["correct"] / row["n"], 6) if row["n"] else None
    return rows


def llm_vs_history(outcomes, actual: Sequence[str]) -> Dict[str, Any]:
    rows = [
        (outcome, truth)
        for outcome, truth in zip(outcomes, actual)
        if outcome.history_verdict is not None
    ]
    llm_correct = sum(outcome.verdict == truth for outcome, truth in rows)
    history_correct = sum(outcome.history_verdict == truth for outcome, truth in rows)
    llm_only = sum(outcome.verdict == truth and outcome.history_verdict != truth for outcome, truth in rows)
    history_only = sum(outcome.verdict != truth and outcome.history_verdict == truth for outcome, truth in rows)
    both = sum(outcome.verdict == truth and outcome.history_verdict == truth for outcome, truth in rows)
    neither = sum(outcome.verdict != truth and outcome.history_verdict != truth for outcome, truth in rows)
    return {
        "n": len(rows),
        "llm_correct": llm_correct,
        "llm_accuracy": round(llm_correct / len(rows), 6) if rows else None,
        "history_correct": history_correct,
        "history_accuracy": round(history_correct / len(rows), 6) if rows else None,
        "llm_only_correct": llm_only,
        "history_only_correct": history_only,
        "both_correct": both,
        "neither_correct": neither,
    }


def n5a_report(results, decisions, actual: Sequence[str]) -> Dict[str, Any]:
    selected = [
        (result, truth)
        for result, decision, truth in zip(results, decisions, actual)
        if decision.branch == "N5a"
    ]
    pure = sum(result.is_label_pure for result, _ in selected)
    majority_correct = sum(
        majority_label([item.label for item in result.top_candidates if item.label is not None]) == truth
        for result, truth in selected
    )
    return {
        "case_count": len(selected),
        "pure_signature_cases": pure,
        "mixed_signature_cases": len(selected) - pure,
        "mixed_signature_ratio": round((len(selected) - pure) / len(selected), 6) if selected else 0.0,
        "bucket_majority_correct": majority_correct,
        "bucket_majority_accuracy": round(majority_correct / len(selected), 6) if selected else None,
    }


def match_record(result) -> Dict[str, Any]:
    """逐 case 只保存参与分支判断的最高分候选，避免 top_k=0 时重复整个训练图。"""
    value = result.to_dict()
    value["retrieved_candidate_count"] = len(result.candidates)
    value["candidates"] = [item.to_dict() for item in result.top_candidates]
    value["dual_candidates"] = [item.to_dict() for item in result.dual_top_candidates]
    return value


def selective_risk_curve(
    outcomes,
    actual: Sequence[str],
    *,
    minimum_support: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for threshold in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        selected = [
            (outcome, truth)
            for outcome, truth in zip(outcomes, actual)
            if outcome.verdict is not None
            and outcome.calibration_support >= minimum_support
            and outcome.confidence_lower_bound >= threshold
        ]
        correct = sum(outcome.verdict == truth for outcome, truth in selected)
        precision = correct / len(selected) if selected else None
        rows.append({
            "minimum_lower_bound": threshold,
            "answered": len(selected),
            "coverage": round(len(selected) / len(actual), 6) if actual else 0.0,
            "correct": correct,
            "precision_at_coverage": round(precision, 6) if precision is not None else None,
            "selective_risk": round(1.0 - precision, 6) if precision is not None else None,
        })
    return rows


def decision_report(
    final_decisions,
    actual: Sequence[str],
    *,
    sop_predictions: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    counts = Counter(item.action for item in final_decisions)
    predictions = [item.verdict for item in final_decisions]
    answered = [(prediction, truth) for prediction, truth in zip(predictions, actual) if prediction is not None]
    correct = sum(prediction == truth for prediction, truth in answered)
    total = len(final_decisions)
    classes = class_metrics(predictions, actual)
    return {
        "actions": {action: counts.get(action, 0) for action in ("final", "request_evidence", "human_review")},
        "answered": len(answered),
        "coverage": round(len(answered) / total, 6) if total else 0.0,
        "correct": correct,
        "precision_when_answered": round(correct / len(answered), 6) if answered else None,
        "coverage_accuracy": round(correct / total, 6) if total else 0.0,
        "low_confidence_degradation_rate": round(
            (counts.get("request_evidence", 0) + counts.get("human_review", 0)) / total, 6
        ) if total else 0.0,
        "human_intervention_rate": round(counts.get("human_review", 0) / total, 6) if total else 0.0,
        "class_metrics": classes,
        "by_candidate_source": candidate_source_report(final_decisions, actual),
        "degeneracy_guard": degeneracy_guard(
            final_decisions, actual, classes, sop_predictions=sop_predictions
        ),
    }


def personal_alignment_gate(decision_policy: DecisionPolicy) -> Dict[str, Any]:
    return {
        "schema_version": "personal-rca-loop-gate-v1",
        "authority": "docs/个人整体思路.md",
        "main_path": (
            "evidence graph match -> N5a historical chain reuse / "
            "N5b physics key-evidence LLM arbitration / "
            "N5c expert SOP constrained LLM -> M9 degradation"
        ),
        "candidate_order": list(decision_policy.candidate_order),
        "candidate_order_ok": tuple(decision_policy.candidate_order) == ("branch",),
        "expert_and_sop_are_advisory": tuple(decision_policy.candidate_order) == ("branch",),
        "expert_sop_version": EXPERT_SOP_VERSION,
        "expert_sop_hash": expert_sop_hash(),
        "n8_feedback_update": False,
        "forbidden_loop_target": "forced three-class accuracy or lift alone without confidence calibration",
    }


def _sop_verdict(prediction: Any) -> Optional[str]:
    """SOP 预测在不同调用点分别是 dataclass 与 dict，这里统一取 verdict。"""
    if prediction is None:
        return None
    if isinstance(prediction, Mapping):
        return prediction.get("verdict")
    return getattr(prediction, "verdict", None)


def degeneracy_guard(
    final_decisions,
    actual: Sequence[str],
    classes: Dict[str, Any],
    *,
    sop_predictions: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """三个指标，专门用来识破「靠多数类蒙对」的退化解。

    `coverage` 与 `precision_when_answered` 有个共同盲区：在 rca_v2_l2fixed 上
    一律报 L2 就有 62.6% 的准确率和 0% 人工干预，两项都好看，但对 L1 与 fiber 毫无用处。
    迭代 1 的嵌套验证进一步显示，把配置选择的代价算进去后 SOP 反而不如直接报多数类
    （lift -2.58pp），所以这三项必须与覆盖率一起看：

    - `lift_over_majority_on_kept`：只在**被保留的那批 case** 上和多数类比。
      必须同子集比，否则测的只是「门限挑走了容易的 case」。
    - `balanced_recall`：三类召回的算术平均，多数类蒙对拉不动它。
    - `abstention_effectiveness`：被弃答的 case 里，若用 SOP 兜底会答错的比例。
      它回答「人工是否被用在了对的地方」——降低人工比例只有在弃答仍然精准时才算改进。
    """
    truths = list(actual)
    counts = Counter(truths)
    majority = max(counts, key=lambda label: counts[label]) if counts else None
    kept = [
        (item, truth)
        for item, truth in zip(final_decisions, truths)
        if item.verdict is not None
    ]
    kept_correct = sum(1 for item, truth in kept if item.verdict == truth)
    majority_on_kept = sum(1 for _, truth in kept if truth == majority)
    recalls = [classes[label]["recall"] for label in ROOT_CAUSES if classes[label]["support"]]

    abstained = [
        index
        for index, item in enumerate(final_decisions)
        if item.verdict is None
    ]
    abstention: Dict[str, Any] = {"abstained": len(abstained)}
    if sop_predictions is not None:
        forced = [
            (index, _sop_verdict(sop_predictions[index]))
            for index in abstained
            if index < len(sop_predictions) and _sop_verdict(sop_predictions[index])
        ]
        wrong = sum(1 for index, verdict in forced if verdict != truths[index])
        abstention.update(
            {
                "with_sop_fallback": len(forced),
                "sop_would_be_wrong": wrong,
                "effectiveness": round(wrong / len(forced), 6) if forced else None,
            }
        )

    return {
        "majority_label": majority,
        "majority_rate_over_all": round(counts.get(majority, 0) / len(truths), 6) if truths else None,
        "majority_on_kept": round(majority_on_kept / len(kept), 6) if kept else None,
        "lift_over_majority_on_kept": (
            round((kept_correct - majority_on_kept) / len(kept), 6) if kept else None
        ),
        "balanced_recall": round(sum(recalls) / len(recalls), 6) if recalls else 0.0,
        "abstention_effectiveness": abstention,
    }


def candidate_source_report(final_decisions, actual: Sequence[str]) -> Dict[str, Any]:
    """按候选来源拆分自动结论。

    没有这张表就无法回答「覆盖率的提升是来自 case 特异证据还是群体先验」，
    而这正是审稿人与运维都会问的第一个问题。
    """
    tally: Dict[str, List[int]] = {}
    for item, truth in zip(final_decisions, actual):
        if item.action != "final" or item.verdict is None:
            continue
        row = tally.setdefault(item.candidate_source, [0, 0])
        row[0] += int(item.verdict == truth)
        row[1] += 1
    return {
        source: {
            "answered": value[1],
            "correct": value[0],
            "precision_when_answered": round(value[0] / value[1], 6) if value[1] else None,
        }
        for source, value in sorted(tally.items())
    }


def fit_train_decision_policy(
    policy,
    graph,
    train_results,
    train_packs,
    train_features,
    train_labels: Sequence[str],
    *,
    sop_model=None,
    target_selective_risk: float,
    minimum_support: int,
    candidate_order: Tuple[str, ...],
    non_identifiable_labels: Tuple[str, ...] = (),
    non_identifiable_evidence: Optional[Mapping[str, Tuple[str, ...]]] = None,
    class_conditional: bool = False,
) -> Tuple[DecisionPolicy, Dict[str, Any]]:
    """在训练留一法输出上反解 M9 工作点。

    分支输出用留一法检索，SOP 候选用留一法重拟合的树，两者都不包含被评估的 case。
    这里刻意不接 LLM：阈值应当由确定性证据链与统计先验定出来，
    否则每次换模型或换 prompt 都要重新标定门禁。
    """
    calibration = fit_calibration(
        train_results, train_packs, train_labels, policy=policy, source="manifest-train-loo"
    )
    paired = handle_many(
        train_results,
        train_packs,
        calibration,
        policy=policy,
        reasoner=None,
        features=train_features,
        sop_model=sop_model,
    )
    outcomes = [item[1] for item in paired]
    # 折外而非留一：留一法下同一叶子里「符合结论」的 case 置信度必然低于
    # 「不符合」的 case，用它反解门限会反向筛选（见 `_out_of_fold_sop_predictions`）。
    oof_sop = (
        _out_of_fold_sop_predictions(train_features, train_labels, sop=sop_model)
        if sop_model is not None
        else [None] * len(outcomes)
    )
    oof_expert = (
        out_of_fold_expert_predictions(train_packs, train_labels)
        if "expert" in candidate_order
        else [None] * len(outcomes)
    )
    probe = DecisionPolicy(
        final_lower_bound=0.0,
        minimum_support=minimum_support,
        candidate_order=candidate_order,
    )
    rows = [
        (
            build_candidates(
                outcome,
                sop_prediction=sop_pred,
                expert_prediction=expert_pred,
                policy=probe,
            ),
            truth,
        )
        for outcome, sop_pred, expert_pred, truth in zip(
            outcomes, oof_sop, oof_expert, train_labels
        )
    ]
    return fit_decision_policy(
        rows,
        target_selective_risk=target_selective_risk,
        minimum_support=minimum_support,
        candidate_order=candidate_order,
        non_identifiable_labels=non_identifiable_labels,
        non_identifiable_evidence=non_identifiable_evidence,
        source=f"manifest-train-loo:{policy.name}",
        class_conditional=class_conditional,
    )


def run_policy(policy, graph, train_results, train_packs, train_labels,
               test_results, test_packs, test_labels, reasoner=None,
               decision_policy: DecisionPolicy = DEFAULT_DECISION_POLICY,
               calibrate_llm: bool = True,
               train_features=None,
               test_features=None,
               sop_model=None,
               branch_calibration: Optional[Any] = None,
               llm_calibration_override: Optional[LLMCalibration] = None,
               expert_calibration: Optional[ExpertCalibration] = None,
               ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    calibration = branch_calibration or fit_calibration(
        train_results, train_packs, train_labels, policy=policy
    )
    llm_calibration = llm_calibration_override
    if reasoner is not None and calibrate_llm and llm_calibration is None:
        calibration_traces: Dict[str, Any] = {}
        calibration_paired = handle_many(
            train_results,
            train_packs,
            calibration,
            policy=policy,
            reasoner=reasoner,
            trace_collector=calibration_traces,
            features=train_features,
            sop_model=sop_model,
        )
        calibration_outcomes = [item[1] for item in calibration_paired]
        llm_calibration = LLMCalibration.fit(
            calibration_outcomes,
            [calibration_traces.get(outcome.case_id) for outcome in calibration_outcomes],
            train_labels,
            source=f"train-loo-llm:{policy.name}",
        )

    traces: Dict[str, Any] = {}
    paired = handle_many(
        test_results,
        test_packs,
        calibration,
        policy=policy,
        reasoner=reasoner,
        llm_calibration=llm_calibration,
        trace_collector=traces,
        features=test_features,
        sop_model=sop_model,
    )
    decisions = [item[0] for item in paired]
    outcomes = [item[1] for item in paired]
    sop_predictions: List[Optional[Dict[str, Any]]] = [
        sop_model.predict(
            numeric_features_from_pack(test_packs[index])
            if getattr(sop_model, "version", "").startswith("numeric-decision-tree")
            else test_features[index]
        ).to_dict()
        if sop_model is not None and test_features is not None
        else None
        for index in range(len(outcomes))
    ]
    # 专家规则在测试期照常运行；被训练边界冻结的是它的可靠性标定，不是规则本身。
    expert_diagnoses = diagnose_many(test_packs)
    expert_predictions: List[Optional[Dict[str, Any]]] = [
        expert_calibration.prediction(diagnosis) if expert_calibration is not None else None
        for diagnosis in expert_diagnoses
    ]
    causal_expert_assessments = [
        assess_filtered_rule_expert(
            expert_group=diagnosis.group,
            expert_verdict=diagnosis.verdict,
            symptom_side=diagnosis.sides[0].side if diagnosis.sides else None,
            tokens=test_features[index].tokens if test_features is not None else (),
            telemetry=test_packs[index].to_dict().get("telemetry", {}),
        )
        if test_packs[index].source_dataset in SOURCE_TOPOLOGIES else None
        for index, diagnosis in enumerate(expert_diagnoses)
    ]
    final_decisions = decide_many(
        outcomes,
        decision_policy,
        sop_predictions=sop_predictions,
        expert_predictions=expert_predictions,
    )

    answered = [(o, t) for o, t in zip(outcomes, test_labels) if o.verdict is not None]
    correct = sum(o.verdict == t for o, t in answered)

    report = {
        "policy": policy.to_dict(),
        "calibration": calibration.to_dict(),
        "llm_calibration": llm_calibration.to_dict() if llm_calibration is not None else None,
        "decision_policy": decision_policy.to_dict(),
        "routing": routing_summary(decisions),
        "n5a": n5a_report(test_results, decisions, test_labels),
        "branches": branch_report(outcomes, decisions, test_labels),
        "calibration_check": calibration_check(outcomes, test_labels),
        "personal_alignment_gate": personal_alignment_gate(decision_policy),
        "answered": len(answered),
        "answer_rate": round(len(answered) / len(outcomes), 6) if outcomes else 0.0,
        "correct": correct,
        "precision_when_answered": round(correct / len(answered), 6) if answered else None,
        "coverage_accuracy": round(correct / len(outcomes), 6) if outcomes else 0.0,
        "raw_class_metrics": class_metrics([outcome.verdict for outcome in outcomes], test_labels),
        "forced_class_metrics": class_metrics([outcome.verdict for outcome in outcomes], test_labels),
        "confidence_reliability": confidence_reliability(outcomes, test_labels),
        "dimension_reliability": dimension_reliability(outcomes, test_labels),
        "threshold_sweep": threshold_sweep(outcomes, test_labels),
        "branch_class_matrix": branch_class_matrix(outcomes, decisions, test_labels),
        "llm_vs_history": llm_vs_history(outcomes, test_labels),
        "selective_risk_curve": selective_risk_curve(
            outcomes, test_labels, minimum_support=decision_policy.minimum_support
        ),
        "final_decisions": decision_report(
            final_decisions, test_labels, sop_predictions=sop_predictions
        ),
    }
    records = []
    for index, (result, routing, outcome, final_decision, truth) in enumerate(
        zip(test_results, decisions, outcomes, final_decisions, test_labels)
    ):
        diagnosis = None
        if test_features is not None:
            diagnosis = build_case_diagnosis(
                test_packs[index],
                test_features[index],
                outcome,
                final_decision,
                sop_version=sop_model.version if sop_model is not None else SOP_VERSION,
                constraint_library_version=(
                    f"{PHYSICS_LIBRARY.version}+{MEASUREMENT_CONTRACT_LIBRARY.version}"
                    if test_packs[index].source_dataset in SOURCE_TOPOLOGIES
                    else CONSTRAINT_LIBRARY.version
                ),
            )
        report_record = build_report(outcome, final_decision, diagnosis=diagnosis).to_dict()
        feature_record = test_features[index].to_dict() if test_features is not None else None
        sop_prediction = sop_predictions[index]
        records.append({
            "case_id": outcome.case_id,
            "actual": truth,
            "evidence_pack": test_packs[index].to_dict(),
            "features": feature_record,
            "sop_prediction": sop_prediction,
            "expert_diagnosis": expert_diagnoses[index].to_dict(),
            "filtered_rule_expert_assessment": (
                causal_expert_assessments[index].to_dict()
                if causal_expert_assessments[index] is not None else None
            ),
            "expert_prediction": expert_predictions[index],
            "match": match_record(result),
            "routing": routing.to_dict(),
            "branch_outcome": outcome.to_dict(),
            "final_decision": final_decision.to_dict(),
            "diagnosis_graph": diagnosis.to_dict() if diagnosis is not None else None,
            "report": report_record,
            "trace_id": outcome.case_id if outcome.case_id in traces else None,
        })
    return report, records, traces


def show(report: Dict[str, Any]) -> None:
    policy = report["policy"]
    print(f"===== 路由规则：{policy['name']} =====")
    print(f"  {policy['description']}\n")

    routing = report["routing"]
    counts = ", ".join(f"{k}={v}" for k, v in routing["counts"].items())
    print(f"  分流分布   : {counts}（共 {routing['case_count']} 条）")
    print(f"  有补采清单 : {routing['cases_with_missing_evidence']} 条\n")

    print(f"  {'分支':<6} {'n':>4} {'给结论':>6} {'判对':>5} {'给结论时准确率':>14} {'需LLM':>6} {'需人工':>6}")
    for branch in ("N5a", "N5b", "N5c", "N6"):
        row = report["branches"].get(branch)
        if not row:
            continue
        precision = row["precision_when_answered"]
        print(
            f"  {branch:<6} {row['n']:>4} {row['answered']:>6} {row['correct']:>5} "
            f"{(f'{precision:.2%}' if precision is not None else '-'):>14} "
            f"{row['needs_llm']:>6} {row['needs_human']:>6}"
        )
    print()

    print(f"  {'标定分组':<20} {'声称置信度':>10} {'95%下界':>9} {'留出实测':>9} {'差距':>8} {'n':>4}")
    for group, row in report["calibration_check"].items():
        observed = row["observed_accuracy"]
        gap = row.get("gap")
        print(
            f"  {group:<20} {row['claimed_confidence']:>10.2%} {row['claimed_lower_bound']:>9.2%} "
            f"{(f'{observed:.2%}' if observed is not None else '-'):>9} "
            f"{(f'{gap:+.2%}' if gap is not None else '-'):>8} {row['answered']:>4}"
        )
    print()
    print(f"  给出结论    : {report['answered']} 条（{report['answer_rate']:.2%}）")
    print(f"  给结论时准确率: {report['correct']}/{report['answered']} = {report['precision_when_answered']:.2%}")
    print(f"  全量准确率  : {report['coverage_accuracy']:.2%}（弃权计为不对，与 legacy 58/85=68.24% 同口径）")
    final = report["final_decisions"]
    final_precision = final["precision_when_answered"]
    print(
        f"  M9 最终出口 : {final['answered']} 条（覆盖率 {final['coverage']:.2%}，"
        f"给结论时准确率 {(f'{final_precision:.2%}' if final_precision is not None else '-')}）"
    )
    print(
        f"  降级分布    : 补采 {final['actions']['request_evidence']}，"
        f"人工 {final['actions']['human_review']}"
    )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/organized_rca_v2_stratified_60_40_seed42"))
    parser.add_argument("--train-size", type=int, default=126)
    parser.add_argument("--manifest-split", action="store_true",
                        help="从 data-dir/_metadata/manifest.json 显式读取 train/test split")
    parser.add_argument("--feature-profile", default="v1", choices=("v1", "v2", "v3", "all_families"),
                        help="特征字典 profile；l2fixed v2 实验应使用 v2")
    parser.add_argument("--learned-sop", action="store_true",
                        help="在训练集上学习 learned-sop-v1，并接入 N5c dry-run")
    parser.add_argument("--output", type=Path, default=None,
                        help="兼容旧调用：只写聚合 JSON；正式实验请用 --output-dir")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="正式实验目录：写 summary/run_manifest/outcomes/traces 四类产物")
    parser.add_argument("--top-k", type=int, default=0, help="历史候选数；0 表示保留全部候选")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policies", nargs="+", default=(BOARD_POLICY.name, COVERAGE_POLICY.name),
                        choices=(BOARD_POLICY.name, COVERAGE_POLICY.name))
    parser.add_argument("--decision-lower-bound", type=float, default=0.5)
    parser.add_argument("--decision-min-support", type=int, default=10)
    parser.add_argument(
        "--target-selective-risk",
        type=float,
        default=None,
        help="给定时忽略 --decision-lower-bound，改为在训练留一法上反解出满足该风险的最大覆盖率工作点",
    )
    parser.add_argument(
        "--class-conditional-bounds",
        action="store_true",
        help=(
            "在统一门限之上按预测类别逐类校准下界，要求每一类的选择性风险各自达标；"
            "单一门限会因类别先验差异结构性地挡掉少数类"
        ),
    )
    parser.add_argument(
        "--decision-candidate-order",
        nargs="+",
        default=("branch",),
        choices=CANDIDATE_SOURCES,
        help=(
            "M9 候选级联顺序。正式默认只接受 branch；加入 sop/expert 仅用于显式"
            "消融或对照实验，不能替代证据图匹配主干"
        ),
    )
    parser.add_argument(
        "--non-identifiable-labels",
        nargs="*",
        default=(),
        choices=ROOT_CAUSES,
        help=(
            "在现有遥测下不可识别的根因（见 C20）。命中的候选不输出结论，"
            "改为带定向补采清单的 request_evidence"
        ),
    )
    parser.add_argument("--skip-llm-calibration", action="store_true",
                        help="仅用于工程冒烟；正式实验默认在训练集留一法输出上标定 LLM")
    parser.add_argument("--llm-backend", default="none", choices=("none", "vllm"),
                        help="三分支仲裁后端。none 表示不调模型，保留确定性分支输出。")
    parser.add_argument("--model-path", default="", help="--llm-backend vllm 时必填")
    parser.add_argument("--max-attempts", type=int, default=2,
                        help="约束校验失败后的最大重写次数，用尽仍不合规则弃权")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="推理型模型的思考段很长，给少了会被截断而解析失败")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--disable-custom-all-reduce", action="store_true",
                        help="无 NVLink 的多卡机器上建议开启，改走标准 NCCL")
    args = parser.parse_args()

    reasoner = None
    if args.llm_backend != "none":
        from rca_framework.llm import ConstrainedReasoner, backend_for

        reasoner = ConstrainedReasoner(
            backend=backend_for(
                args.llm_backend,
                model_path=args.model_path,
                tensor_parallel_size=args.tensor_parallel_size,
                max_new_tokens=args.max_new_tokens,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                disable_custom_all_reduce=args.disable_custom_all_reduce,
                seed=args.seed,
            ),
            max_attempts=args.max_attempts,
        )

    split_manifest = None
    if args.manifest_split:
        split_manifest = load_split_manifest(args.data_dir)
        train_cases = cases_by_manifest_split(args.data_dir, "train")
        test_cases = cases_by_manifest_split(args.data_dir, "test")
    else:
        cases = load_cases(args.data_dir)
        train_cases, test_cases = cases[: args.train_size], cases[args.train_size :]
    dictionary = dictionary_for(args.feature_profile)
    thresholds = fit_thresholds(train_cases)
    train_packs = build_packs(train_cases, source_dataset=str(args.data_dir))
    test_packs = build_packs(test_cases, source_dataset=str(args.data_dir))
    model = fit_feature_model(train_packs, dictionary=dictionary)
    train_features = [extract_features(pack, thresholds, model, dictionary=dictionary) for pack in train_packs]
    test_features = [extract_features(pack, thresholds, model, dictionary=dictionary) for pack in test_packs]
    sop_model = learn_sop(
        train_features,
        labels_of(train_cases),
        source=f"{args.data_dir.name}:manifest-train" if args.manifest_split else f"{args.data_dir.name}:positional-train",
    ) if args.learned_sop else None
    graph = EvidenceGraph.build(
        train_features, labels_of(train_cases), feature_model=model, dictionary=dictionary, source_dataset=str(args.data_dir)
    )

    train_results = match_many(graph, train_features, top_k=args.top_k, leave_one_out=True)
    test_results = match_many(graph, test_features, top_k=args.top_k)

    print(f"graph_version : {graph.version}")
    print(f"train / test  : {len(train_cases)} / {len(test_cases)}")
    print(f"features      : {dictionary.version} ({dictionary.content_hash()})")
    if sop_model is not None:
        print(f"sop           : {sop_model.version} ({sop_model.content_hash()})")
    print(f"llm backend   : {args.llm_backend}"
          + ("（不执行 LLM 仲裁）" if args.llm_backend == "none" else f"，最多重写 {args.max_attempts} 次") + "\n")

    policy_by_name = {BOARD_POLICY.name: BOARD_POLICY, COVERAGE_POLICY.name: COVERAGE_POLICY}
    candidate_order = tuple(args.decision_candidate_order)
    non_identifiable = tuple(args.non_identifiable_labels)
    non_identifiable_evidence = {
        label: FIBER_EVIDENCE_REQUEST for label in non_identifiable if label == "fiber"
    }
    reports: Dict[str, Any] = {}
    all_records: Dict[str, Any] = {}
    all_traces: Dict[str, Any] = {}
    decision_fits: Dict[str, Any] = {}
    for policy_name in args.policies:
        policy = policy_by_name[policy_name]
        decision_policy = DecisionPolicy(
            final_lower_bound=args.decision_lower_bound,
            minimum_support=args.decision_min_support,
            candidate_order=candidate_order,
            non_identifiable_labels=non_identifiable,
            non_identifiable_evidence=non_identifiable_evidence,
        )
        if args.target_selective_risk is not None:
            decision_policy, fit = fit_train_decision_policy(
                policy,
                graph,
                train_results,
                train_packs,
                train_features,
                labels_of(train_cases),
                sop_model=sop_model,
                target_selective_risk=args.target_selective_risk,
                minimum_support=args.decision_min_support,
                candidate_order=candidate_order,
                non_identifiable_labels=non_identifiable,
                non_identifiable_evidence=non_identifiable_evidence,
                class_conditional=args.class_conditional_bounds,
            )
            decision_fits[policy_name] = fit
            print(f"decision fit  : {decision_policy.fitted_on}\n")
        report, records, traces = run_policy(
            policy, graph, train_results, train_packs, labels_of(train_cases),
            test_results, test_packs, labels_of(test_cases), reasoner=reasoner,
            decision_policy=decision_policy,
            calibrate_llm=not args.skip_llm_calibration,
            train_features=train_features,
            test_features=test_features,
            sop_model=sop_model,
            expert_calibration=ExpertCalibration.fit(
                diagnose_many(train_packs), labels_of(train_cases), source="manifest-train"
            ),
        )
        reports[policy.name] = report
        all_records[policy.name] = records
        all_traces[policy.name] = {
            case_id: trace.to_dict() for case_id, trace in sorted(traces.items())
        }
        show(report)

    manifest = {
        "schema_version": "agentic-rca-run-manifest-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "data": {
            "data_dir": str(args.data_dir),
            "train_size": len(train_cases),
            "test_size": len(test_cases),
            "train_case_ids": [pack.case_id for pack in train_packs],
            "test_case_ids": [pack.case_id for pack in test_packs],
            "split_seed": args.seed,
            "split_manifest_schema": split_manifest.get("schema_version") if split_manifest else None,
            "split_manifest_hash": split_manifest.get("source_hash") if split_manifest else None,
        },
        "versions": {
            "evidence_graph": graph.version,
            "feature_dictionary": graph.dictionary_version,
            "feature_dictionary_hash": graph.dictionary_hash,
            "constraint_library": CONSTRAINT_LIBRARY.version,
            "constraint_library_hash": CONSTRAINT_LIBRARY.content_hash(),
            "sop": sop_model.version if sop_model is not None else SOP_VERSION,
            "sop_hash": sop_model.content_hash() if sop_model is not None else None,
            "expert_sop": EXPERT_SOP_VERSION,
            "expert_sop_hash": expert_sop_hash(),
            "prompt_template": PROMPT_TEMPLATE_VERSION,
            "prompt_template_hash": prompt_template_hash(),
            "decision_policy": decision_policy.version,
        },
        "retrieval": {
            "top_k": args.top_k,
            "routing_policies": [policy_by_name[name].to_dict() for name in args.policies],
        },
        "decision": decision_policy.to_dict(),
        "personal_alignment_gate": personal_alignment_gate(decision_policy),
        "scope": {
            "self_evolution": False,
            "feedback_update": False,
            "loop_target": (
                "evidence-graph shape, evidence-chain/path matching, physics key-evidence "
                "judgment, or expert-SOP-constrained LLM reasoning"
            ),
        },
        "llm": {
            "backend": args.llm_backend,
            "model_path": args.model_path,
            "max_attempts": args.max_attempts,
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_new_tokens": args.max_new_tokens,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "disable_custom_all_reduce": args.disable_custom_all_reduce,
            "temperature": 0.0,
            "seed": args.seed,
            "calibration": (
                "train-loo confidence-bin reliability"
                if reasoner is not None and not args.skip_llm_calibration
                else "disabled"
            ),
        },
        "training_graph_purity": graph.purity_report(),
        "learned_sop": sop_model.to_dict() if sop_model is not None else None,
        "leakage_policy": (
            "阈值、特征模型、证据图、分支置信度与 LLM 可靠性标定只使用训练集；"
            "留出测试标签仅在全部推理完成后用于指标计算。"
        ),
    }
    payload = {
        "graph_version": graph.version,
        "manifest": manifest,
        "policies": reports,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.output}")
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        artifacts: Mapping[str, Any] = {
            "summary.json": {"graph_version": graph.version, "policies": reports},
            "run_manifest.json": manifest,
            "outcomes.json": all_records,
            "traces.json": all_traces,
        }
        for filename, value in artifacts.items():
            path = args.output_dir / filename
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
