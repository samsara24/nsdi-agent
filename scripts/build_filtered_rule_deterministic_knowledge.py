#!/usr/bin/env python3
"""Build and audit filtered-rule train knowledge without calling an LLM."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca_framework.data import cases_by_manifest_split, split_manifest_hash  # noqa: E402
from rca_framework.decision_tree import numeric_features_from_packs  # noqa: E402
from rca_framework.evidence_graph import (  # noqa: E402
    FILTERED_RULE_THREE_CHANNEL_POLICY,
    match_many,
    route_many,
)
from rca_framework.evidence_pack import build_packs, labels_of  # noqa: E402
from rca_framework.knowledge import fit_offline_knowledge  # noqa: E402


DEFAULT_DATA_DIR = Path("datasets/filtered_rule_temporal_2025_06_09_v1")
DEFAULT_OUTPUT_DIR = Path("artifacts/filtered_rule_deterministic_knowledge_v1")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_report(path: Path, summary: Mapping[str, Any], artifacts: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]) -> None:
    sop_correct = sum(
        row["learned_sop_prediction"].get("verdict") == row["label"] for row in cases
    )
    fiber_rows = [row for row in cases if row["label"] == "fiber"]
    fiber_correct = sum(
        row["learned_sop_prediction"].get("verdict") == "fiber" for row in fiber_rows
    )
    purity = artifacts["graph_purity"]
    lines = [
        "# Filtered-rule 确定性训练知识审计",
        "",
        "本目录仅由固定训练 split 生成，不调用 LLM，不读取测试标签，也不执行 N8 回灌。",
        "",
        "## 已沉淀资产",
        "",
        f"- 训练 case：{summary['train_case_count']} 条；来源分布 `{summary['source_distribution']}`。",
        f"- 可解释 token：{summary['token_count']} 种。",
        f"- 证据图：{summary['evidence_graph_version']}，包含 {summary['evidence_graph_case_count']} 个历史 case。",
        f"- learned SOP：{summary['learned_sop_version']}，hash `{summary['learned_sop_hash']}`。",
        f"- 留一法路由：`{summary['routing_distribution']}`。",
        "- 每条训练 case 的特征、数值量测、SOP 路径、留一法路由和 Top-5 历史候选保存在 `case_audit.json`。",
        "",
        "## 可复核结论",
        "",
        f"- {purity['signature_group_count']} 个 signature 中 {purity['singleton_group_count']} 个仅有 1 条支持；"
        "训练 signature 高纯度不能直接解释为可泛化准确率。",
        f"- 混合标签 signature 覆盖 {purity['mixed_label_case_count']} 条，不能作为 N5a 自动复用模式。",
        f"- 数值 learned SOP 训练内命中 {sop_correct}/{len(cases)}；fiber 命中 {fiber_correct}/{len(fiber_rows)}。"
        "该树只能作为统计先验，不能作为 fiber 或端点归因的物理证据。",
        "- SerDes SNR 数值尺度尚未完成量测语义确认；树中相关分位数切分只保留审计用途。",
        "- LLM calibration 与 LLM trace 均为空。正式测试应在加载本知识包后才调用 LLM。",
        "",
        "## 文件",
        "",
        "- `knowledge_bundle.json`：可重新加载的训练知识包。",
        "- `training_summary.json`：图纯度、SOP、分支与决策阈值。",
        "- `case_audit.json`：124 条逐 case 审计。",
        "- `signature_audit.json`：完整 signature 分组和标签纯度。",
        "- `token_audit.json`：每个可解释 token 的支持数和标签分布。",
        "- `audit_summary.json`：版本、hash 和关键计数。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _signature_key(tokens: Iterable[str]) -> str:
    return "\n".join(sorted(tokens))


def _signature_audit(features: Sequence[Any], labels: Sequence[str]) -> Dict[str, Any]:
    groups: Dict[str, list[tuple[str, str]]] = defaultdict(list)
    for feature, label in zip(features, labels):
        groups[_signature_key(feature.tokens)].append((feature.case_id, label))
    rows = []
    mixed_case_count = 0
    for signature, members in sorted(groups.items()):
        counts = Counter(label for _, label in members)
        pure = len(counts) == 1
        if not pure:
            mixed_case_count += len(members)
        rows.append({
            "signature_tokens": signature.splitlines() if signature else [],
            "case_ids": [case_id for case_id, _ in members],
            "label_distribution": dict(sorted(counts.items())),
            "support": len(members),
            "pure": pure,
        })
    return {
        "signature_count": len(rows),
        "pure_signature_count": sum(row["pure"] for row in rows),
        "mixed_signature_count": sum(not row["pure"] for row in rows),
        "mixed_signature_case_count": mixed_case_count,
        "empty_signature_case_count": sum(
            row["support"] for row in rows if not row["signature_tokens"]
        ),
        "groups": rows,
    }


def _token_audit(features: Sequence[Any], labels: Sequence[str]) -> Dict[str, Any]:
    support: Counter[str] = Counter()
    by_label: Dict[str, Counter[str]] = defaultdict(Counter)
    for feature, label in zip(features, labels):
        for token in feature.tokens:
            support[token] += 1
            by_label[token][label] += 1
    return {
        "token_count": len(support),
        "tokens": [
            {
                "token": token,
                "support": count,
                "label_distribution": dict(sorted(by_label[token].items())),
                "label_purity": round(max(by_label[token].values()) / count, 6),
            }
            for token, count in sorted(support.items())
        ],
    }


def _case_audit(
    cases: Sequence[Mapping[str, Any]],
    packs: Sequence[Any],
    features: Sequence[Any],
    numeric_rows: Sequence[Any],
    results: Sequence[Any],
    routes: Sequence[Any],
    sop: Any,
) -> list[Dict[str, Any]]:
    rows = []
    for case, pack, feature, numeric, result, decision in zip(
        cases, packs, features, numeric_rows, results, routes
    ):
        rows.append({
            "case_id": pack.case_id,
            "label": case["label"],
            "original_label": case.get("original_label"),
            "source_dataset": pack.source_dataset,
            "topology_id": pack.topology_id,
            "lane_profile": pack.lane_profile,
            "telemetry_status": pack.telemetry_status,
            "coverage": pack.coverage,
            "missing_fields": list(pack.missing_fields),
            "optical_blackout": pack.optical_blackout,
            "explainable_features": feature.to_dict(),
            "numeric_features": numeric.to_dict(),
            "learned_sop_prediction": sop.predict(numeric).to_dict(),
            "leave_one_out_route": decision.to_dict(),
            "leave_one_out_history": [
                candidate.to_dict() for candidate in result.candidates[:5]
            ],
        })
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-train-size", type=int, default=124)
    return parser


def main() -> None:
    args = _parser().parse_args()
    train_cases = cases_by_manifest_split(args.data_dir, "train")
    if len(train_cases) != args.expected_train_size:
        raise ValueError(
            f"expected {args.expected_train_size} train cases, got {len(train_cases)}"
        )
    labels = labels_of(train_cases)
    policy = FILTERED_RULE_THREE_CHANNEL_POLICY
    source_id = args.data_dir.name
    bundle, artifacts = fit_offline_knowledge(
        train_cases,
        source_dataset=source_id,
        split_manifest_hash=split_manifest_hash(args.data_dir),
        feature_profile="filtered_rule_v1",
        policies=(policy,),
        reasoner=None,
        top_k=0,
        target_selective_risk=0.15,
        decision_minimum_support=10,
        decision_candidate_order=("branch",),
        build_metadata={
            "knowledge_build_mode": "deterministic-train-only-v1",
            "llm_calls": 0,
            "label_leakage": False,
            "n8_frozen": True,
        },
    )
    if bundle.llm_calibrations:
        raise RuntimeError("deterministic build unexpectedly produced LLM calibration")
    if any(artifacts.traces.values()):
        raise RuntimeError("deterministic build unexpectedly produced LLM traces")

    packs = tuple(build_packs(train_cases, source_dataset=source_id))
    numeric_rows = numeric_features_from_packs(packs)
    results = match_many(bundle.graph, bundle.training_features, top_k=0, leave_one_out=True)
    routes = route_many(results, policy)
    signature = _signature_audit(bundle.training_features, labels)
    token = _token_audit(bundle.training_features, labels)
    cases = _case_audit(
        train_cases, packs, bundle.training_features, numeric_rows, results, routes, bundle.sop
    )
    route_counts = Counter(item.branch for item in routes)
    source_counts = Counter(pack.source_dataset for pack in packs)
    label_counts = Counter(labels)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = bundle.save(args.output_dir / "knowledge_bundle.json")
    _write_json(args.output_dir / "training_summary.json", artifacts.summary)
    _write_json(args.output_dir / "case_audit.json", cases)
    _write_json(args.output_dir / "signature_audit.json", signature)
    _write_json(args.output_dir / "token_audit.json", token)
    summary = {
        "schema_version": "filtered-rule-deterministic-knowledge-audit-v1",
        "knowledge_build_mode": "deterministic-train-only-v1",
        "llm_calls": 0,
        "train_case_count": len(train_cases),
        "source_distribution": dict(sorted(source_counts.items())),
        "label_distribution": dict(sorted(label_counts.items())),
        "routing_distribution": dict(sorted(route_counts.items())),
        "feature_dictionary_version": bundle.feature_model.dictionary_version,
        "feature_dictionary_hash": bundle.feature_model.dictionary_hash,
        "feature_model_version": bundle.feature_model.version,
        "evidence_graph_version": bundle.graph.version,
        "evidence_graph_case_count": len(bundle.graph.cases),
        "evidence_graph_diagnosis_count": len(bundle.graph.case_diagnoses),
        "learned_sop_version": bundle.sop.version,
        "learned_sop_hash": bundle.sop.content_hash(),
        "knowledge_bundle_hash": bundle.content_hash(),
        "knowledge_bundle_path": str(bundle_path),
        "signature_summary": {
            key: value for key, value in signature.items() if key != "groups"
        },
        "token_count": token["token_count"],
        "llm_calibrations": 0,
        "llm_traces": 0,
        "label_leakage": False,
        "n8_frozen": True,
    }
    _write_json(args.output_dir / "audit_summary.json", summary)
    _write_report(args.output_dir / "README.md", summary, artifacts.summary, cases)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
