"""Dual-similarity routing and executable SOP for expanded expert-clean RCA.

This module is independent from the legacy KG/fusion classifier.  It keeps
feature and observable-graph similarity separate, calibrates a conservative
history-reuse gate on train-only leave-one-out data, and compiles raw telemetry
into an auditable Q0/P/R/L SOP trace.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .anomaly import METRIC_ALIASES, ThresholdModel, fit_thresholds, metric_values
from .evidence_pack import EvidencePack, build_packs
from .expanded_evidence import (
    case_quality_state,
    fit_edge_idf,
    physical_evidence_paths,
    quality_compatible,
    weighted_edge_jaccard,
)
from .features.dictionary import dictionary_for
from .features.extractor import CaseFeatures, extract_features, fit_feature_model
from .evidence_graph.match import find_conflicts, weighted_jaccard
from .types import ROOT_CAUSES


DUAL_MATCH_VERSION = "expanded-dual-match-v1"
ROUTING_POLICY_VERSION = "expanded-dual-routing-v1"
SOP_EXECUTOR_VERSION = "expanded-executable-sop-v1"
DEFAULT_TOP_K = 5
DEFAULT_MIN_JOINT_CANDIDATES = 3
DEFAULT_MAX_SELECTIVE_RISK = 0.15
DEFAULT_MIN_CALIBRATION_SUPPORT = 20
THRESHOLD_GRID = tuple(round(0.70 + 0.05 * index, 2) for index in range(7))
SOP_STEP_IDS = (
    "Q0_validate_measurements",
    "P_apply_physical_boundaries",
    "R_expand_directional_chain",
    "L_apply_stable_learned_ranges",
    "D_select_or_request_evidence",
)


@dataclass(frozen=True)
class LearnedPredicate:
    predicate_id: str
    side: str
    metric: str
    aggregation: str
    operator: str
    threshold: float
    target: str
    support: int
    purity: float
    provenance: str = "learned-predicate-ranges-v1:clean-train-122"

    def evaluate(self, case: Mapping[str, Any]) -> tuple[bool, Optional[float]]:
        values = metric_values(dict(case), self.metric, self.side, healthy_only=True)
        if not values:
            return False, None
        if self.aggregation == "min":
            value = min(values)
        elif self.aggregation == "spread":
            value = max(values) - min(values)
        else:
            raise ValueError(f"unsupported aggregation: {self.aggregation}")
        matched = value <= self.threshold if self.operator == "<=" else value > self.threshold
        return matched, float(value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicate_id": self.predicate_id, "side": self.side, "metric": self.metric,
            "aggregation": self.aggregation, "operator": self.operator,
            "threshold": self.threshold, "target": self.target, "support": self.support,
            "purity": self.purity, "provenance": self.provenance,
        }


LEARNED_PREDICATES: Tuple[LearnedPredicate, ...] = (
    LearnedPredicate(
        "L1_l2_media_snr_min_le_23_805", "L2", "media_snr", "min", "<=",
        23.805, "L1", 10, 0.8,
    ),
    LearnedPredicate(
        "L2_l2_rxpower_spread_gt_2_165", "L2", "rxpower", "spread", ">",
        2.165, "L1", 10, 1.0,
    ),
)


def _edge(src: str, relation: str, dst: str) -> str:
    return f"{src}|{relation}|{dst}"


def path_edges(path: Mapping[str, Any]) -> Tuple[str, ...]:
    return (
        _edge(str(path["side"]), "HAS_MEASUREMENT", str(path["measurement"])),
        _edge(str(path["measurement"]), "SATISFIES", str(path["predicate"])),
        _edge(str(path["predicate"]), "INDICATES", str(path["symptom"])),
        _edge(str(path["symptom"]), "LOCATED_AT", str(path["layer"])),
    )


def _learned_paths(case: Mapping[str, Any]) -> Tuple[Dict[str, Any], ...]:
    rows = []
    for item in LEARNED_PREDICATES:
        matched, value = item.evaluate(case)
        if not matched:
            continue
        predicate = f"predicate:{item.side}:{item.metric}:{item.predicate_id}"
        rows.append({
            "side": f"side:{item.side}",
            "measurement": f"measurement:{item.side}:{item.metric}",
            "predicate": predicate,
            "symptom": f"symptom:learned_range_supports_{item.target}",
            "layer": "physical-layer:statistical_candidate",
            "token": f"learned:{item.predicate_id}",
            "predicate_type": "training_learned_range",
            "criterion": f"observed {value:.6g} {item.operator} {item.threshold:.6g}",
            "provenance": item.provenance,
            "quantifier": item.aggregation,
            "learned": True,
            "observed_value": value,
            "target": item.target,
        })
    return tuple(rows)


def _augment_features(features: CaseFeatures, case: Mapping[str, Any]) -> CaseFeatures:
    physical = {str(path["token"]) for path in physical_evidence_paths(dict(case))}
    learned = {str(path["token"]) for path in _learned_paths(case)}
    features.tokens = tuple(sorted(set(features.tokens) | physical | learned))
    features.by_family = dict(features.by_family)
    features.by_family["expanded_physical_state"] = tuple(sorted(physical))
    features.by_family["expanded_learned_range"] = tuple(sorted(learned))
    features.dictionary_version = "expanded-explainable-features-v2"
    return features


@dataclass(frozen=True)
class EvidenceView:
    case_id: str
    feature_tokens: Tuple[str, ...]
    graph_edges: Tuple[str, ...]
    paths: Tuple[Dict[str, Any], ...]
    quality: str
    missing_measurements: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id, "feature_tokens": list(self.feature_tokens),
            "graph_edges": list(self.graph_edges), "paths": list(self.paths),
            "quality": self.quality, "missing_measurements": list(self.missing_measurements),
        }


def build_views(
    cases: Sequence[Dict[str, Any]],
    *,
    thresholds: Optional[ThresholdModel] = None,
    feature_model: Any = None,
) -> tuple[Tuple[EvidenceView, ...], ThresholdModel, Any, Tuple[EvidencePack, ...]]:
    packs = tuple(build_packs(cases, source_dataset="expanded-expert-clean-v1"))
    fitted_thresholds = thresholds or fit_thresholds(cases)
    dictionary = dictionary_for("v1")
    fitted_feature_model = feature_model or fit_feature_model(packs, dictionary=dictionary)
    views = []
    for case, pack in zip(cases, packs):
        features = _augment_features(
            extract_features(pack, fitted_thresholds, fitted_feature_model, dictionary=dictionary), case
        )
        paths = tuple(physical_evidence_paths(case)) + _learned_paths(case)
        edges = tuple(sorted({edge for path in paths for edge in path_edges(path)}))
        quality = case_quality_state(case)
        quality_name = str(quality["quality"])
        if quality["measurements"] and all(
            state["observed"] == 0 for state in quality["measurements"].values()
        ):
            quality_name = "no_valid_telemetry"
        views.append(EvidenceView(
            case_id=str(case["case_id"]), feature_tokens=features.tokens,
            graph_edges=edges, paths=paths, quality=quality_name,
            missing_measurements=tuple(quality["missing_measurements"]),
        ))
    return tuple(views), fitted_thresholds, fitted_feature_model, packs


@dataclass(frozen=True)
class DualCandidate:
    case_id: str
    label: str
    feature_similarity: float
    graph_similarity: float
    quality_compatible: bool
    critical_conflicts: Tuple[Tuple[str, str], ...] = ()
    missing_feature_evidence: Tuple[str, ...] = ()
    missing_graph_edges: Tuple[str, ...] = ()

    @property
    def rank_key(self) -> tuple[float, float, str]:
        return (-min(self.feature_similarity, self.graph_similarity),
                -(self.feature_similarity + self.graph_similarity), self.case_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id, "label": self.label,
            "feature_similarity": self.feature_similarity,
            "graph_similarity": self.graph_similarity,
            "quality_compatible": self.quality_compatible,
            "critical_conflicts": [list(item) for item in self.critical_conflicts],
            "missing_feature_evidence": list(self.missing_feature_evidence),
            "missing_graph_edges": list(self.missing_graph_edges),
        }


@dataclass(frozen=True)
class DualMatchResult:
    query_case_id: str
    query_view: EvidenceView
    candidates: Tuple[DualCandidate, ...]
    feature_threshold: float
    graph_threshold: float
    top_k: int = DEFAULT_TOP_K
    min_joint_candidates: int = DEFAULT_MIN_JOINT_CANDIDATES

    @property
    def joint_candidates(self) -> Tuple[DualCandidate, ...]:
        eligible = [item for item in self.candidates if (
            item.feature_similarity >= self.feature_threshold
            and item.graph_similarity >= self.graph_threshold
            and item.quality_compatible
        )]
        return tuple(sorted(eligible, key=lambda item: item.rank_key)[: self.top_k])

    @property
    def label_distribution(self) -> Dict[str, int]:
        return dict(Counter(item.label for item in self.joint_candidates))

    @property
    def label_pure(self) -> bool:
        return len(self.label_distribution) == 1

    @property
    def has_critical_conflict(self) -> bool:
        return any(item.critical_conflicts for item in self.joint_candidates)

    @property
    def strict_reuse(self) -> bool:
        return (
            len(self.joint_candidates) >= self.min_joint_candidates
            and self.label_pure
            and not self.has_critical_conflict
        )

    @property
    def max_feature_similarity(self) -> float:
        return max((item.feature_similarity for item in self.candidates), default=0.0)

    @property
    def max_graph_similarity(self) -> float:
        return max((item.graph_similarity for item in self.candidates), default=0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": DUAL_MATCH_VERSION, "query_case_id": self.query_case_id,
            "feature_similarity": self.max_feature_similarity,
            "graph_similarity": self.max_graph_similarity,
            "feature_threshold": self.feature_threshold, "graph_threshold": self.graph_threshold,
            "joint_candidate_count": len(self.joint_candidates),
            "label_distribution": self.label_distribution, "label_pure": self.label_pure,
            "has_critical_conflict": self.has_critical_conflict, "strict_reuse": self.strict_reuse,
            "query_paths": list(self.query_view.paths),
            "candidates": [item.to_dict() for item in self.candidates[: self.top_k]],
            "joint_candidates": [item.to_dict() for item in self.joint_candidates],
        }


def _idf(rows: Iterable[Iterable[str]]) -> Dict[str, float]:
    sets = [set(row) for row in rows]
    frequency = Counter(token for row in sets for token in row)
    return {token: math.log((1 + len(sets)) / (1 + count)) + 1.0 for token, count in frequency.items()}


class DualMatcher:
    def __init__(self, train_cases: Sequence[Dict[str, Any]], train_views: Sequence[EvidenceView]):
        self.train_cases = tuple(train_cases)
        self.train_views = tuple(train_views)
        self.feature_idf = _idf(view.feature_tokens for view in train_views)
        self.graph_idf = fit_edge_idf({"edges": view.graph_edges} for view in train_views)

    def match(
        self,
        case: Dict[str, Any],
        view: EvidenceView,
        *,
        feature_threshold: float,
        graph_threshold: float,
        exclude_case_id: str = "",
    ) -> DualMatchResult:
        rows = []
        query_tokens, query_edges = set(view.feature_tokens), set(view.graph_edges)
        for historical, historical_view in zip(self.train_cases, self.train_views):
            case_id = str(historical["case_id"])
            if case_id == exclude_case_id:
                continue
            history_tokens, history_edges = set(historical_view.feature_tokens), set(historical_view.graph_edges)
            rows.append(DualCandidate(
                case_id=case_id, label=str(historical["label"]),
                feature_similarity=weighted_jaccard(query_tokens, history_tokens, self.feature_idf),
                graph_similarity=weighted_edge_jaccard(query_edges, history_edges, self.graph_idf),
                quality_compatible=quality_compatible(case, historical),
                critical_conflicts=find_conflicts(query_tokens, history_tokens),
                missing_feature_evidence=tuple(sorted(history_tokens - query_tokens)),
                missing_graph_edges=tuple(sorted(history_edges - query_edges)),
            ))
        rows.sort(key=lambda item: item.rank_key)
        return DualMatchResult(
            query_case_id=str(case["case_id"]), query_view=view, candidates=tuple(rows),
            feature_threshold=feature_threshold, graph_threshold=graph_threshold,
        )


@dataclass(frozen=True)
class CalibratedDualPolicy:
    feature_threshold: float
    graph_threshold: float
    support: int
    correct: int
    precision: float
    fallback_exact: bool
    curve: Tuple[Dict[str, Any], ...] = ()
    version: str = ROUTING_POLICY_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version, "feature_threshold": self.feature_threshold,
            "graph_threshold": self.graph_threshold, "support": self.support,
            "correct": self.correct, "precision": self.precision,
            "fallback_exact": self.fallback_exact, "max_selective_risk": DEFAULT_MAX_SELECTIVE_RISK,
            "minimum_support": DEFAULT_MIN_CALIBRATION_SUPPORT,
            "minimum_joint_candidates": DEFAULT_MIN_JOINT_CANDIDATES,
            "top_k": DEFAULT_TOP_K, "curve": list(self.curve),
        }


def calibrate_dual_policy(
    train_cases: Sequence[Dict[str, Any]], train_views: Sequence[EvidenceView], matcher: DualMatcher,
) -> CalibratedDualPolicy:
    curve = []
    qualified = []
    for feature_threshold in THRESHOLD_GRID:
        for graph_threshold in THRESHOLD_GRID:
            answered = correct = 0
            for case, view in zip(train_cases, train_views):
                result = matcher.match(
                    case, view, feature_threshold=feature_threshold,
                    graph_threshold=graph_threshold, exclude_case_id=str(case["case_id"]),
                )
                if not result.strict_reuse:
                    continue
                answered += 1
                verdict = next(iter(result.label_distribution))
                correct += int(verdict == case["label"])
            precision = correct / answered if answered else 0.0
            row = {
                "feature_threshold": feature_threshold, "graph_threshold": graph_threshold,
                "support": answered, "correct": correct, "precision": round(precision, 8),
            }
            curve.append(row)
            if answered >= DEFAULT_MIN_CALIBRATION_SUPPORT and precision >= 1 - DEFAULT_MAX_SELECTIVE_RISK:
                qualified.append(row)
    if qualified:
        chosen = max(qualified, key=lambda row: (
            row["support"], row["precision"], row["feature_threshold"] + row["graph_threshold"]
        ))
        fallback = False
    else:
        exact = next(row for row in curve if row["feature_threshold"] == 1.0 and row["graph_threshold"] == 1.0)
        chosen, fallback = exact, True
    return CalibratedDualPolicy(
        feature_threshold=float(chosen["feature_threshold"]),
        graph_threshold=float(chosen["graph_threshold"]), support=int(chosen["support"]),
        correct=int(chosen["correct"]), precision=float(chosen["precision"]),
        fallback_exact=fallback, curve=tuple(curve),
    )


@dataclass(frozen=True)
class RoutingDecision:
    branch: str
    reason: str

    def to_dict(self) -> Dict[str, str]:
        return {"branch": self.branch, "reason": self.reason, "policy": ROUTING_POLICY_VERSION}


def route_dual(result: DualMatchResult) -> RoutingDecision:
    view = result.query_view
    if view.quality == "optical_blackout":
        return RoutingDecision("N6", "Q0 optical blackout：哨兵表示量测无效，转人工现场确认")
    if view.quality == "no_valid_telemetry":
        return RoutingDecision("N6", "Q0 无有效遥测：直接请求补采或人工介入，不交给模型猜测")
    if not view.feature_tokens and not view.graph_edges:
        return RoutingDecision("N6", "没有可用特征或证据路径，无法推理")
    if result.strict_reuse:
        return RoutingDecision("N5a", "双维阈值通过、Top-K 标签纯净、质量兼容且无关键冲突")
    feature_high = result.max_feature_similarity >= result.feature_threshold
    graph_high = result.max_graph_similarity >= result.graph_threshold
    if feature_high or graph_high or result.joint_candidates:
        return RoutingDecision("N5b", "仅部分满足双维历史门禁，需检查差异、关键缺失与标签冲突")
    return RoutingDecision("N5c", "两个相似度都低于训练集校准门槛，进入未见模式 SOP")


@dataclass(frozen=True)
class SOPStepResult:
    step_id: str
    status: str
    statement: str
    evidence_ids: Tuple[str, ...] = ()
    candidate_root_causes: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id, "status": self.status, "statement": self.statement,
            "evidence_ids": list(self.evidence_ids),
            "candidate_root_causes": list(self.candidate_root_causes),
        }


@dataclass(frozen=True)
class SOPResult:
    steps: Tuple[SOPStepResult, ...]
    candidates: Tuple[str, ...]
    deterministic_verdict: Optional[str]
    decision_action: str
    missing_information: Tuple[str, ...]
    forced_prediction: str
    version: str = SOP_EXECUTOR_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version, "steps": [item.to_dict() for item in self.steps],
            "candidates": list(self.candidates), "deterministic_verdict": self.deterministic_verdict,
            "decision_action": self.decision_action,
            "missing_information": list(self.missing_information),
            "forced_prediction": self.forced_prediction,
        }


def raw_measurement_snapshot(case: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"units": {
        "txpower": "dBm", "rxpower": "dBm", "media_snr": "field-unit",
        "host_snr": "field-unit", "serdes_snr": "unknown/binary-validity-only",
        "Temperature": "degC", "Voltage": "V",
    }}
    for side in ("L1", "L2"):
        result[side] = {
            metric: list(metric_values(dict(case), metric, side)) for metric in sorted(METRIC_ALIASES)
        }
        result[side]["Lane number"] = (case.get("Lane number") or {}).get(side)
        for name in ("RxLOS", "RxLOL", "TxLOS", "TxLOL", "Temperature", "Voltage", "port_status"):
            value = case.get(name)
            result[side][name] = value.get(side) if isinstance(value, dict) else value
    return result


def execute_sop(case: Dict[str, Any], view: EvidenceView, *, majority_label: str = "L2") -> SOPResult:
    steps = []
    quality = case_quality_state(case)
    if view.quality == "no_valid_telemetry":
        steps.append(SOPStepResult(
            SOP_STEP_IDS[0], "blocked", "无有效遥测，禁止历史复用、统计推断或模型猜测", (), (),
        ))
        return SOPResult(tuple(steps), tuple(ROOT_CAUSES), None, "request_evidence",
                         ("重新采集双端光功率、SNR 与端口状态",), majority_label)
    if quality["quality"] == "optical_blackout":
        steps.append(SOPStepResult(
            SOP_STEP_IDS[0], "blocked", "双端光学读数触底且 SNR 无效，不能把 -40 解释为未发光",
            tuple(path["token"] for path in view.paths), (),
        ))
        return SOPResult(tuple(steps), tuple(ROOT_CAUSES), None, "human_review",
                         ("现场确认模块与采集链路状态",), majority_label)
    steps.append(SOPStepResult(
        SOP_STEP_IDS[0], "passed" if quality["quality"] == "valid" else "partial",
        f"量测质量={quality['quality']}，缺失 {len(quality['missing_measurements'])} 个指标",
        tuple(f"quality:{item}" for item in quality["missing_measurements"]), tuple(ROOT_CAUSES),
    ))

    endpoint_votes: Counter[str] = Counter()
    ambiguous = False
    evidence_ids = []
    for path in view.paths:
        token = str(path["token"])
        evidence_ids.append(token)
        side = str(path["side"]).split(":", 1)[-1]
        if "txpower:exact_minus_40" in token or "txpower:value_le_minus_39" in token:
            endpoint_votes[side] += 3
        elif "rx_present:decode_invalid" in token or "serdes_snr:value_le_1" in token:
            endpoint_votes[side] += 2
        elif token.startswith("relation:") and ":tx_present:" in token and ":rx_down" in token:
            ambiguous = True
        elif token.startswith("learned:"):
            predicate = next(item for item in LEARNED_PREDICATES if item.predicate_id in token)
            endpoint_votes[predicate.target] += 1
    steps.append(SOPStepResult(
        SOP_STEP_IDS[1], "evaluated", "先应用 -40/-39/0/1 物理边界；物理边界优先于学习范围",
        tuple(evidence_ids), tuple(sorted(endpoint_votes) or ROOT_CAUSES),
    ))
    relation_candidates = set(endpoint_votes)
    if ambiguous:
        relation_candidates.update(ROOT_CAUSES)
    steps.append(SOPStepResult(
        SOP_STEP_IDS[2], "ambiguous" if ambiguous else "evaluated",
        "接收方向异常展开为对端发送链、介质和本端接收链；未用跨端 lane 功率相减",
        tuple(token for token in evidence_ids if token.startswith("relation:")),
        tuple(label for label in ROOT_CAUSES if label in relation_candidates) or tuple(ROOT_CAUSES),
    ))
    learned_ids = tuple(token for token in evidence_ids if token.startswith("learned:"))
    steps.append(SOPStepResult(
        SOP_STEP_IDS[3], "matched" if learned_ids else "not_matched",
        f"仅执行 {len(LEARNED_PREDICATES)} 条训练集稳定谓词，命中 {len(learned_ids)} 条",
        learned_ids, tuple(sorted(endpoint_votes) or ROOT_CAUSES),
    ))

    winners: Tuple[str, ...] = ()
    if endpoint_votes:
        best = max(endpoint_votes.values())
        winners = tuple(label for label in ROOT_CAUSES if endpoint_votes[label] == best)
    decisive = len(winners) == 1 and not ambiguous and winners[0] != "fiber"
    verdict = winners[0] if decisive else None
    candidates = winners or tuple(ROOT_CAUSES)
    missing = []
    if ambiguous:
        missing.extend(("可信双向功率标定", "对端发送链与本端接收链自检结果"))
    if "fiber" in candidates or not candidates:
        missing.extend(("OTDR 曲线", "端面镜检或换纤复测"))
    if quality["missing_measurements"]:
        missing.append("补采关键缺失遥测：" + ", ".join(quality["missing_measurements"]))
    action = "automatic_conclusion" if verdict else "request_evidence"
    forced = verdict or (winners[0] if winners else majority_label)
    steps.append(SOPStepResult(
        SOP_STEP_IDS[4], "decisive" if verdict else "insufficient",
        "形成唯一端点候选" if verdict else "现有证据不能唯一排除候选，生产口径请求补采",
        (), candidates,
    ))
    return SOPResult(tuple(steps), candidates, verdict, action, tuple(dict.fromkeys(missing)), forced)


_THRESHOLD_PATTERN = re.compile(r"(?:<=|>=|<|>|低于|高于|不超过|超过)\s*(-?\d+(?:\.\d+)?)")


def validate_expanded_llm_response(response: Any, request: Any) -> Tuple[str, ...]:
    """Additional expanded-only checks layered on the shared constraint checker."""
    violations = []
    allowed_steps = set(SOP_STEP_IDS)
    declared = {-40.0, -39.0, 0.0, 1.0, 23.805, 2.165}
    sop_candidates = set(getattr(request, "sop_candidates", ()) or ROOT_CAUSES)
    step_order = {step_id: index for index, step_id in enumerate(SOP_STEP_IDS)}
    allowed_evidence = set(getattr(request, "evidence_tokens", ()))
    allowed_predicates = {
        str(item.get("predicate_id")) for item in getattr(request, "declared_predicates", ())
        if item.get("predicate_id")
    }
    last_order = -1
    for index, step in enumerate(getattr(response, "steps", ())):
        step_id = getattr(step, "sop_step_id", "")
        if step_id not in allowed_steps:
            violations.append(f"step {index} 缺少合法 sop_step_id")
        elif step_order[step_id] < last_order:
            violations.append(f"step {index} 违反 Q0→P→R→L→D 顺序")
        else:
            last_order = step_order[step_id]
        cited_evidence = set(getattr(step, "cited_evidence", ()))
        cited_predicates = set(getattr(step, "cited_predicates", ()))
        if not cited_evidence:
            violations.append(f"step {index} 未引用 evidence ID")
        elif not cited_evidence <= allowed_evidence:
            violations.append(f"step {index} 引用了不存在的 evidence ID")
        if not cited_predicates:
            violations.append(f"step {index} 未引用 predicate ID")
        elif not cited_predicates <= allowed_predicates:
            violations.append(f"step {index} 引用了未声明 predicate ID")
        for match in _THRESHOLD_PATTERN.finditer(getattr(step, "claim", "")):
            value = float(match.group(1))
            if not any(math.isclose(value, item, abs_tol=1e-6) for item in declared):
                violations.append(f"step {index} 使用未声明阈值 {value}")
    verdict = getattr(response, "verdict", None)
    if verdict not in sop_candidates:
        violations.append(f"verdict {verdict} 不在 SOP 候选 {sorted(sop_candidates)} 中")
    if verdict == "fiber":
        violations.append("缺少现场介质证据，fiber 不得成为生产自动结论")
    return tuple(violations)
