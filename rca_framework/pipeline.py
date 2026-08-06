from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from . import graph as graph_module
from . import rules as rules_module
from .anomaly import ThresholdModel, extract_evidence, fit_thresholds
from .evidence import AGREEMENT_TYPES, EvidenceView, aggregate_evidence
from .fusion import fuse_results
from .graph import AnomalyKnowledgeGraph, CoverageReport
from .llm import PathLLMReasoner
from .rules import SymbolicRuleEngine
from .runtime import RuntimeConfig
from .types import CaseEvidence, ROOT_CAUSES


LEAKAGE_GUARD_NOTE = "The target label was removed before anomaly extraction, graph query, retrieval, prompting and rule matching."


@dataclass
class PipelineConfig:
    min_edge_count: int = 1
    min_rule_count: int = 2
    min_rule_confidence: float = 0.35
    min_rule_lift: float = 1.05
    min_rule_margin: float = 0.03
    max_rules_per_class: int = 40
    top_k_paths: int = 12
    top_k_cases: int = 5
    graph_weight: float = 0.55
    symbolic_weight: float = 0.45
    conflict_dominance_gap: float = 0.20
    manual_review_margin: float = 0.10


@dataclass
class CaseContext:
    """单个 case 在调用 LLM 之前已经装配好的确定性上下文。

    `coverage` 与 `evidence_view` 是阶段 1 引入的观测量，legacy 决策不读取它们。
    """

    case_id: str
    actual_label: str
    evidence: CaseEvidence
    graph_result: Dict[str, Any]
    symbolic_result: Dict[str, Any]
    coverage: CoverageReport
    evidence_view: EvidenceView

    def observation(self) -> Dict[str, Any]:
        """纯观测字段。改变这里不会改变 legacy 的 prediction 与 scores。"""
        return {
            "coverage": self.coverage.to_dict(),
            "score_composition": self.graph_result.get("score_composition", {}),
            "evidence": self.evidence_view.to_dict(),
            "support_tier_counts": self.symbolic_result.get("support_tier_counts", {}),
        }


def build_case_context(
    case: Dict[str, Any],
    thresholds: ThresholdModel,
    graph: AnomalyKnowledgeGraph,
    rules: SymbolicRuleEngine,
    config: PipelineConfig,
) -> CaseContext:
    """提取异常、查询图谱、匹配符号规则，并在此之前先摘掉标签。"""
    target = dict(case)
    actual_label = str(target.pop("label", ""))
    evidence = extract_evidence(target, thresholds)
    graph_result = graph.query(evidence, config.top_k_paths, config.top_k_cases)
    symbolic_result = rules.match(evidence)
    return CaseContext(
        case_id=evidence.case_id,
        actual_label=actual_label,
        evidence=evidence,
        graph_result=graph_result,
        symbolic_result=symbolic_result,
        coverage=graph_module.classify_coverage(evidence, graph_result),
        evidence_view=aggregate_evidence(
            graph_module.evidence_items(graph_result) + rules_module.evidence_items(symbolic_result)
        ),
    )


def finalize_prediction(context: CaseContext, llm_result: Dict[str, Any], config: PipelineConfig) -> Dict[str, Any]:
    """把图谱侧观测挂回 LLM 路结果，再做 legacy 两路融合并组装输出记录。"""
    graph_result = context.graph_result
    llm_result["graph_paths"] = graph_result["paths"]
    llm_result["matched_kg_feature_rules"] = graph_result.get("matched_feature_rules", {})
    llm_result["feature_profile_scores"] = graph_result.get("feature_profile_scores", {})
    llm_result["retrieved_cases"] = graph_result["retrieved_cases"]
    llm_result["evidence_coverage"] = graph_result["evidence_coverage"]
    final = fuse_results(
        context.evidence,
        llm_result,
        context.symbolic_result,
        graph_weight=config.graph_weight,
        symbolic_weight=config.symbolic_weight,
        dominance_gap=config.conflict_dominance_gap,
        review_margin=config.manual_review_margin,
    )
    return {
        "case_id": context.case_id,
        "prediction": final["prediction"],
        "confidence": final["confidence"],
        "decision_status": final["decision_status"],
        "final_decision": final,
        "KG_RAG_LLM": llm_result,
        "KG_RCA": context.symbolic_result,
        "extracted_anomalies": [item.to_dict() for item in context.evidence.anomalies],
        "leakage_guard": LEAKAGE_GUARD_NOTE,
        "observation": context.observation(),
    }


class RCAPipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.thresholds: ThresholdModel | None = None
        self.graph = AnomalyKnowledgeGraph()
        self.rules = SymbolicRuleEngine()
        self.training_case_ids: List[str] = []
        self._reasoners: Dict[RuntimeConfig, PathLLMReasoner] = {}

    def fit(self, cases: Sequence[Dict[str, Any]]) -> "RCAPipeline":
        labeled = [case for case in cases if case.get("label") in ROOT_CAUSES]
        if not labeled:
            raise ValueError("training requires L1/L2/fiber labels")
        self.thresholds = fit_thresholds(labeled)
        views = [extract_evidence(case, self.thresholds) for case in labeled]
        self.graph.fit(views, min_edge_count=self.config.min_edge_count)
        self.rules.fit(
            views,
            min_count=self.config.min_rule_count,
            min_confidence=self.config.min_rule_confidence,
            min_lift=self.config.min_rule_lift,
            min_margin=self.config.min_rule_margin,
            max_rules_per_class=self.config.max_rules_per_class,
        )
        self.training_case_ids = [view.case_id for view in views]
        return self

    def build_context(self, case: Dict[str, Any]) -> CaseContext:
        if self.thresholds is None:
            raise RuntimeError("pipeline is not fitted")
        return build_case_context(case, self.thresholds, self.graph, self.rules, self.config)

    def _resolve_runtime(self, runtime: RuntimeConfig | None, settings: Dict[str, Any]) -> RuntimeConfig:
        if runtime is not None and settings:
            raise ValueError("pass either runtime or individual runtime settings, not both")
        return runtime if runtime is not None else RuntimeConfig.from_kwargs(settings)

    def _reasoner(self, runtime: RuntimeConfig) -> PathLLMReasoner:
        if runtime not in self._reasoners:
            self._reasoners[runtime] = PathLLMReasoner(
                runtime.llm_backend, runtime.model_path, runtime.max_new_tokens,
                runtime.tensor_parallel_size, runtime.gpu_memory_utilization,
                runtime.max_model_len, runtime.dtype, runtime.enforce_eager,
                runtime.guided_json, runtime.disable_custom_all_reduce,
            )
        return self._reasoners[runtime]

    def infer(
        self,
        case: Dict[str, Any],
        *,
        runtime: RuntimeConfig | None = None,
        **runtime_settings: Any,
    ) -> Dict[str, Any]:
        resolved = self._resolve_runtime(runtime, runtime_settings)
        context = self.build_context(case)
        llm_result = self._reasoner(resolved).reason(context.evidence, context.graph_result)
        return finalize_prediction(context, llm_result, self.config)

    def evaluate(
        self,
        cases: Sequence[Dict[str, Any]],
        *,
        runtime: RuntimeConfig | None = None,
        **runtime_settings: Any,
    ) -> Dict[str, Any]:
        resolved = self._resolve_runtime(runtime, runtime_settings)
        contexts = [self.build_context(case) for case in cases]
        llm_rows = self._reasoner(resolved).reason_many(
            [context.evidence for context in contexts],
            [context.graph_result for context in contexts],
        )

        rows = []
        confusion: Dict[str, Counter[str]] = defaultdict(Counter)
        llm_mode_counts: Counter[str] = Counter()
        for context, llm_result in zip(contexts, llm_rows):
            actual = context.actual_label
            result = finalize_prediction(context, llm_result, self.config)
            result["actual_label"] = actual
            result["correct"] = result["prediction"] == actual
            rows.append(result)
            llm_mode_counts[llm_result.get("reasoning_mode", "unknown")] += 1
            if actual in ROOT_CAUSES:
                confusion[actual][result["prediction"]] += 1
        correct = sum(bool(row["correct"]) for row in rows)
        recalls = {}
        for label in ROOT_CAUSES:
            total = sum(confusion[label].values())
            recalls[label] = confusion[label][label] / total if total else None
        return {
            "summary": {
                "case_count": len(rows),
                "correct": correct,
                "accuracy": correct / len(rows) if rows else None,
                "recall": recalls,
                "confusion_matrix": {label: dict(confusion[label]) for label in ROOT_CAUSES},
                "decision_status": dict(Counter(row["decision_status"] for row in rows)),
                "llm_reasoning_mode": dict(llm_mode_counts),
                "valid_llm_outputs": llm_mode_counts["llm_path_reasoning"],
                "label_leakage": False,
                # 阶段 1 观测量。legacy 键的值不受它影响，可直接与历史 summary 比对。
                "observations": self.observation_summary(contexts),
            },
            "predictions": rows,
        }

    def observation_summary(self, contexts: Sequence[CaseContext]) -> Dict[str, Any]:
        """把逐 case 观测量汇总成可直接引用的统计口径。"""
        coverage_states: Counter[str] = Counter()
        agreement_types: Counter[str] = Counter()
        independent_counts: Counter[int] = Counter()
        support_tiers: Counter[str] = Counter()
        prior_only = 0
        exemplar_reachable = 0
        for context in contexts:
            coverage_states[context.coverage.state] += 1
            agreement_types[context.evidence_view.agreement_type] += 1
            independent_counts[context.evidence_view.independent_evidence_count] += 1
            prior_only += int(context.coverage.prior_only)
            exemplar_reachable += int(
                context.coverage.max_retrieval_similarity >= graph_module.EXEMPLAR_SIMILARITY_THRESHOLD
            )
            for tier, count in (context.symbolic_result.get("support_tier_counts") or {}).items():
                support_tiers[tier] += count
        return {
            "coverage_state": {state: coverage_states[state] for state in graph_module.COVERAGE_STATES},
            "prior_only_cases": prior_only,
            "agreement_type": {name: agreement_types[name] for name in AGREEMENT_TYPES},
            "independent_evidence_count": {
                str(key): independent_counts[key] for key in sorted(independent_counts)
            },
            "matched_rule_support_tier": {tier: support_tiers[tier] for tier in rules_module.SUPPORT_TIERS},
            "rule_support_audit": self.rules.support_audit(),
            "retrieval": {
                "exemplar_similarity_threshold": graph_module.EXEMPLAR_SIMILARITY_THRESHOLD,
                "cases_at_or_above_threshold": exemplar_reachable,
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        if self.thresholds is None:
            raise RuntimeError("pipeline is not fitted")
        return {
            "schema": "rca-framework-v2",
            "config": asdict(self.config),
            "thresholds": self.thresholds.to_dict(),
            "knowledge_graph": self.graph.to_dict(),
            "symbolic_rules": self.rules.to_dict(),
            "training_case_ids": self.training_case_ids,
        }

    def save(self, output_dir: Path) -> None:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty model directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        model = self.to_dict()
        (output_dir / "model.json").write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output_dir / "knowledge_graph.json").write_text(json.dumps(model["knowledge_graph"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output_dir / "symbolic_rules.json").write_text(json.dumps(model["symbolic_rules"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output_dir / "thresholds.json").write_text(json.dumps(model["thresholds"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, model_path: Path) -> "RCAPipeline":
        path = model_path / "model.json" if model_path.is_dir() else model_path
        value = json.loads(path.read_text(encoding="utf-8"))
        pipeline = cls(PipelineConfig(**value["config"]))
        pipeline.thresholds = ThresholdModel.from_dict(value["thresholds"])
        pipeline.graph = AnomalyKnowledgeGraph.from_dict(value["knowledge_graph"])
        pipeline.rules = SymbolicRuleEngine.from_dict(value["symbolic_rules"])
        pipeline.training_case_ids = list(value.get("training_case_ids", []))
        return pipeline
