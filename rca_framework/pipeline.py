from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .anomaly import ThresholdModel, extract_evidence, fit_thresholds
from .fusion import fuse_results
from .graph import AnomalyKnowledgeGraph
from .llm import PathLLMReasoner
from .rules import SymbolicRuleEngine
from .types import ROOT_CAUSES


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


class RCAPipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.thresholds: ThresholdModel | None = None
        self.graph = AnomalyKnowledgeGraph()
        self.rules = SymbolicRuleEngine()
        self.training_case_ids: List[str] = []
        self._reasoners: Dict[tuple[Any, ...], PathLLMReasoner] = {}

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

    def infer(
        self,
        case: Dict[str, Any],
        *,
        llm_backend: str = "none",
        model_path: str = "",
        max_new_tokens: int = 512,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 8192,
        dtype: str = "auto",
        enforce_eager: bool = False,
        guided_json: bool = True,
        disable_custom_all_reduce: bool = False,
    ) -> Dict[str, Any]:
        if self.thresholds is None:
            raise RuntimeError("pipeline is not fitted")
        target = dict(case)
        target.pop("label", None)
        evidence = extract_evidence(target, self.thresholds)
        graph_result = self.graph.query(evidence, self.config.top_k_paths, self.config.top_k_cases)
        reasoner_key = (llm_backend, model_path, max_new_tokens, tensor_parallel_size, gpu_memory_utilization, max_model_len, dtype, enforce_eager, guided_json, disable_custom_all_reduce)
        if reasoner_key not in self._reasoners:
            self._reasoners[reasoner_key] = PathLLMReasoner(
                llm_backend, model_path, max_new_tokens, tensor_parallel_size,
                gpu_memory_utilization, max_model_len, dtype, enforce_eager, guided_json,
                disable_custom_all_reduce,
            )
        method1 = self._reasoners[reasoner_key].reason(evidence, graph_result)
        method1["graph_paths"] = graph_result["paths"]
        method1["matched_kg_feature_rules"] = graph_result.get("matched_feature_rules", {})
        method1["feature_profile_scores"] = graph_result.get("feature_profile_scores", {})
        method1["retrieved_cases"] = graph_result["retrieved_cases"]
        method1["evidence_coverage"] = graph_result["evidence_coverage"]
        method2 = self.rules.match(evidence)
        final = fuse_results(
            evidence,
            method1,
            method2,
            graph_weight=self.config.graph_weight,
            symbolic_weight=self.config.symbolic_weight,
            dominance_gap=self.config.conflict_dominance_gap,
            review_margin=self.config.manual_review_margin,
        )
        return {
            "case_id": evidence.case_id,
            "prediction": final["prediction"],
            "confidence": final["confidence"],
            "decision_status": final["decision_status"],
            "final_decision": final,
            "KG_RAG_LLM": method1,
            "KG_RCA": method2,
            "extracted_anomalies": [item.to_dict() for item in evidence.anomalies],
            "leakage_guard": "The target label was removed before anomaly extraction, graph query, retrieval, prompting and rule matching.",
        }

    def evaluate(self, cases: Sequence[Dict[str, Any]], **infer_kwargs: Any) -> Dict[str, Any]:
        if self.thresholds is None:
            raise RuntimeError("pipeline is not fitted")
        prepared = []
        for case in cases:
            target = dict(case)
            actual = str(target.pop("label", ""))
            evidence = extract_evidence(target, self.thresholds)
            graph_result = self.graph.query(evidence, self.config.top_k_paths, self.config.top_k_cases)
            method2 = self.rules.match(evidence)
            prepared.append((actual, evidence, graph_result, method2))

        llm_backend = infer_kwargs.get("llm_backend", "none")
        model_path = infer_kwargs.get("model_path", "")
        max_new_tokens = int(infer_kwargs.get("max_new_tokens", 512))
        tensor_parallel_size = int(infer_kwargs.get("tensor_parallel_size", 1))
        gpu_memory_utilization = float(infer_kwargs.get("gpu_memory_utilization", 0.90))
        max_model_len = int(infer_kwargs.get("max_model_len", 8192))
        dtype = str(infer_kwargs.get("dtype", "auto"))
        enforce_eager = bool(infer_kwargs.get("enforce_eager", False))
        guided_json = bool(infer_kwargs.get("guided_json", True))
        disable_custom_all_reduce = bool(infer_kwargs.get("disable_custom_all_reduce", False))
        reasoner_key = (llm_backend, model_path, max_new_tokens, tensor_parallel_size, gpu_memory_utilization, max_model_len, dtype, enforce_eager, guided_json, disable_custom_all_reduce)
        if reasoner_key not in self._reasoners:
            self._reasoners[reasoner_key] = PathLLMReasoner(
                llm_backend, model_path, max_new_tokens, tensor_parallel_size,
                gpu_memory_utilization, max_model_len, dtype, enforce_eager, guided_json,
                disable_custom_all_reduce,
            )
        method1_rows = self._reasoners[reasoner_key].reason_many(
            [item[1] for item in prepared], [item[2] for item in prepared],
        )

        rows = []
        confusion: Dict[str, Counter[str]] = defaultdict(Counter)
        llm_mode_counts: Counter[str] = Counter()
        for (actual, evidence, graph_result, method2), method1 in zip(prepared, method1_rows):
            method1["graph_paths"] = graph_result["paths"]
            method1["matched_kg_feature_rules"] = graph_result.get("matched_feature_rules", {})
            method1["feature_profile_scores"] = graph_result.get("feature_profile_scores", {})
            method1["retrieved_cases"] = graph_result["retrieved_cases"]
            method1["evidence_coverage"] = graph_result["evidence_coverage"]
            final = fuse_results(
                evidence, method1, method2,
                graph_weight=self.config.graph_weight,
                symbolic_weight=self.config.symbolic_weight,
                dominance_gap=self.config.conflict_dominance_gap,
                review_margin=self.config.manual_review_margin,
            )
            result = {
                "case_id": evidence.case_id,
                "prediction": final["prediction"],
                "confidence": final["confidence"],
                "decision_status": final["decision_status"],
                "final_decision": final,
                "KG_RAG_LLM": method1,
                "KG_RCA": method2,
                "extracted_anomalies": [item.to_dict() for item in evidence.anomalies],
                "leakage_guard": "The target label was removed before anomaly extraction, graph query, retrieval, prompting and rule matching.",
            }
            result["actual_label"] = actual
            result["correct"] = result["prediction"] == actual
            rows.append(result)
            llm_mode_counts[method1.get("reasoning_mode", "unknown")] += 1
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
            },
            "predictions": rows,
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
