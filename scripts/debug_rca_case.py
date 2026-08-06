#!/usr/bin/env python3
"""Inspect one RCA case stage by stage.

This script is intentionally separate from the production CLI.  It exposes
the intermediate anomaly, KG/RAG, symbolic-rule, and fusion results so a new
maintainer can test either reasoning channel without reading predictions.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca_framework.anomaly import extract_evidence
from rca_framework.fusion import fuse_results
from rca_framework.llm import PathLLMReasoner
from rca_framework.pipeline import RCAPipeline


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def attach_graph_context(method: dict[str, Any], graph_result: dict[str, Any]) -> dict[str, Any]:
    method["graph_paths"] = graph_result["paths"]
    method["matched_kg_feature_rules"] = graph_result.get("matched_feature_rules", {})
    method["feature_profile_scores"] = graph_result.get("feature_profile_scores", {})
    method["retrieved_cases"] = graph_result["retrieved_cases"]
    method["evidence_coverage"] = graph_result["evidence_coverage"]
    return method


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Show how one case becomes anomalies, KG/RAG evidence, rules, and a fused RCA result",
    )
    parser.add_argument("--model", required=True, type=Path, help="saved model directory or model.json")
    parser.add_argument("--case", required=True, type=Path, help="one anonymized schema-v2 case")
    parser.add_argument(
        "--channel",
        choices=("evidence", "kg-rag", "kg-rca", "full"),
        default="full",
        help="stop after one stage/channel; full runs both channels and fusion",
    )
    parser.add_argument("--output", type=Path, help="optional full JSON output path")
    parser.add_argument("--backend", choices=("none", "vllm", "transformers"), default="none")
    parser.add_argument("--model-path", default="", help="local LLM path when backend is not none")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--top-k-paths", type=int, default=12)
    parser.add_argument("--top-k-cases", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pipeline = RCAPipeline.load(args.model)
    if pipeline.thresholds is None:
        raise RuntimeError("loaded model has no fitted thresholds")

    raw_case = json.loads(args.case.read_text(encoding="utf-8"))
    reference_label = str(raw_case.get("label", ""))
    target = dict(raw_case)
    target.pop("label", None)
    evidence = extract_evidence(target, pipeline.thresholds)

    report: dict[str, Any] = {
        "case_file": str(args.case),
        "case_id": evidence.case_id,
        "reference_label_for_evaluation_only": reference_label,
        "leakage_guard": "reference label was removed before every inference stage",
        "channel": args.channel,
        "evidence": evidence.to_dict(),
    }

    if args.channel == "evidence":
        result = report
    else:
        if args.channel in {"kg-rag", "full"}:
            graph_result = pipeline.graph.query(evidence, args.top_k_paths, args.top_k_cases)
            reasoner = PathLLMReasoner(
                backend=args.backend,
                model_path=args.model_path,
                max_new_tokens=args.max_new_tokens,
            )
            method1 = attach_graph_context(reasoner.reason(evidence, graph_result), graph_result)
            report["kg_rag_llm"] = method1

        if args.channel in {"kg-rca", "full"}:
            method2 = pipeline.rules.match(evidence)
            report["kg_rca"] = method2

        if args.channel == "full":
            report["fusion"] = fuse_results(
                evidence,
                method1,
                method2,
                graph_weight=pipeline.config.graph_weight,
                symbolic_weight=pipeline.config.symbolic_weight,
                dominance_gap=pipeline.config.conflict_dominance_gap,
                review_margin=pipeline.config.manual_review_margin,
            )
        result = report

    if args.output:
        dump_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
