#!/usr/bin/env python3
"""Audit train/test feature and evidence-graph drift without invoking an LLM."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca_framework.data import cases_by_manifest_split, split_manifest_hash  # noqa: E402
from rca_framework.evidence_graph import (  # noqa: E402
    FILTERED_RULE_THREE_CHANNEL_POLICY,
    match_many,
    route_many,
)
from rca_framework.knowledge import OfflineKnowledgeBundle  # noqa: E402


SPLITS = {
    "test_all_data": "test/all_data",
    "test_rule1_channel_not_4": "test/rule1_channel_not_4",
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_report(path: Path, output: Mapping[str, Any]) -> None:
    lines = [
        "# Filtered-rule train/test 分布审计",
        "",
        "本报告不调用 LLM。特征、图匹配和路由冻结后，测试标签才用于离线统计。",
        "",
    ]
    for name, row in output["splits"].items():
        lines.extend([
            f"## {name}",
            "",
            f"- 测试 case：{row['case_count']}；同来源训练参考：{row['source_conditioned_train_reference_count']}。",
            f"- 训练标签分布：`{row['source_conditioned_train_label_distribution']}`；测试标签分布：`{row['label_distribution_evaluation_only']}`。",
            f"- 双相似度精确匹配：{row['exact_dual_match_count']}/{row['case_count']}；路由：`{row['routing_distribution']}`。",
            f"- S_feature 中位数 {row['feature_similarity']['median']:.3f}，S_graph 中位数 {row['graph_similarity']['median']:.3f}。",
            f"- 最近历史标签直接复用准确率：{row['top1_history_label_accuracy_evaluation_only']:.2%}。",
            f"- 训练未见 token：{row['unseen_train_token_count']} 种；零双重叠 case：{row['zero_dual_overlap_count']}。",
            "",
        ])
    lines.extend([
        "## 结论",
        "",
        "- 两个测试集都不是完全脱离训练图，但精确历史模式覆盖很低，不能把近邻标签当最终结论。",
        "- `rule1_channel_not_4` 存在显著时间 schema 漂移：测试侧 SerDes 系统性缺失。缺测 token 应降低证据完整度，不能投根因票。",
        "- 两个来源均存在标签先验漂移；统一训练池可以共享物理证据，但最终阈值和报告必须按来源分层。",
        "- N5c 占比高是数据分布事实。LLM 应基于当前物理证据独立判断，历史与 learned SOP 只能作为低权重上下文。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _quantiles(values: Sequence[float]) -> Dict[str, float]:
    rows = sorted(values)
    if not rows:
        return {key: 0.0 for key in ("min", "p25", "median", "p75", "max", "mean")}

    def at(fraction: float) -> float:
        return rows[round((len(rows) - 1) * fraction)]

    return {
        "min": round(rows[0], 6),
        "p25": round(at(0.25), 6),
        "median": round(at(0.5), 6),
        "p75": round(at(0.75), 6),
        "max": round(rows[-1], 6),
        "mean": round(mean(rows), 6),
    }


def _token_rates(features: Sequence[Any]) -> Dict[str, float]:
    counts = Counter(token for feature in features for token in set(feature.tokens))
    total = len(features)
    return {token: count / total for token, count in counts.items()} if total else {}


def _bernoulli_js(left: float, right: float) -> float:
    midpoint = (left + right) / 2

    def kl(value: float, target: float) -> float:
        out = 0.0
        for p, q in ((value, target), (1 - value, 1 - target)):
            if p > 0 and q > 0:
                out += p * math.log2(p / q)
        return out

    return (kl(left, midpoint) + kl(right, midpoint)) / 2


def _distribution(counts: Counter[str]) -> Dict[str, int]:
    return dict(sorted(counts.items()))


def _split_audit(
    name: str,
    cases: Sequence[Mapping[str, Any]],
    features: Sequence[Any],
    results: Sequence[Any],
    routes: Sequence[Any],
    train_features: Sequence[Any],
) -> Dict[str, Any]:
    test_sources = {feature.source_dataset for feature in features}
    source_train_features = [
        feature for feature in train_features if feature.source_dataset in test_sources
    ]
    # Compare each isolated test source against its own historical topology. A
    # combined-train comparison would mistake known 4x4/8x8 schema differences
    # for temporal drift.
    train_rates = _token_rates(source_train_features)
    test_rates = _token_rates(features)
    token_union = sorted(set(train_rates) | set(test_rates))
    drift = sorted(
        (
            {
                "token": token,
                "train_rate": round(train_rates.get(token, 0.0), 6),
                "test_rate": round(test_rates.get(token, 0.0), 6),
                "absolute_rate_gap": round(abs(train_rates.get(token, 0.0) - test_rates.get(token, 0.0)), 6),
                "js_divergence": round(_bernoulli_js(train_rates.get(token, 0.0), test_rates.get(token, 0.0)), 6),
            }
            for token in token_union
        ),
        key=lambda row: (-row["js_divergence"], -row["absolute_rate_gap"], row["token"]),
    )
    labels = [str(case["label"]) for case in cases]
    top1 = [
        result.retrieval_candidates[0].label if result.retrieval_candidates else None
        for result in results
    ]
    exact = [
        result.max_feature_similarity == 1.0 and result.max_graph_similarity == 1.0
        for result in results
    ]
    unseen = sorted(set(test_rates) - set(train_rates))
    zero_overlap = [
        result.query_case_id for result in results
        if result.max_feature_similarity == 0.0 and result.max_graph_similarity == 0.0
    ]
    return {
        "split": name,
        "case_count": len(cases),
        "source_distribution": _distribution(Counter(feature.source_dataset for feature in features)),
        "source_conditioned_train_reference_count": len(source_train_features),
        "label_distribution_evaluation_only": _distribution(Counter(labels)),
        "telemetry_status_distribution": _distribution(Counter(feature.telemetry_status for feature in features)),
        "feature_token_count": len(test_rates),
        "unseen_train_token_count": len(unseen),
        "unseen_train_tokens": unseen,
        "token_count_per_case": _quantiles([float(len(feature.tokens)) for feature in features]),
        "feature_similarity": _quantiles([result.max_feature_similarity for result in results]),
        "graph_similarity": _quantiles([result.max_graph_similarity for result in results]),
        "exact_dual_match_count": sum(exact),
        "zero_dual_overlap_count": len(zero_overlap),
        "zero_dual_overlap_case_ids": zero_overlap,
        "cross_topology_fallback_count": sum(result.uses_cross_topology_fallback for result in results),
        "routing_distribution": _distribution(Counter(route.branch for route in routes)),
        "top1_history_label_accuracy_evaluation_only": round(
            sum(predicted == label for predicted, label in zip(top1, labels)) / len(labels), 6
        ) if labels else 0.0,
        "top1_history_missing_count": sum(predicted is None for predicted in top1),
        "top_token_drift": drift[:30],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/filtered_rule_temporal_2025_06_09_v1"))
    parser.add_argument(
        "--knowledge-bundle",
        type=Path,
        default=Path("artifacts/filtered_rule_deterministic_knowledge_v1/knowledge_bundle.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/filtered_rule_deterministic_knowledge_v1/distribution_audit"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    bundle = OfflineKnowledgeBundle.load(args.knowledge_bundle)
    manifest_hash = split_manifest_hash(args.data_dir)
    if bundle.split_manifest_hash != manifest_hash:
        raise ValueError("knowledge bundle and data manifest hashes differ")
    reports: Dict[str, Any] = {}
    for name, split in SPLITS.items():
        cases = cases_by_manifest_split(args.data_dir, split)
        _, features = bundle.extract_test_features(cases, source_dataset=args.data_dir.name)
        results = match_many(bundle.graph, features, top_k=0)
        routes = route_many(results, FILTERED_RULE_THREE_CHANNEL_POLICY)
        reports[name] = _split_audit(
            name, cases, features, results, routes, bundle.training_features
        )
        sources = {feature.source_dataset for feature in features}
        reports[name]["source_conditioned_train_label_distribution"] = _distribution(
            Counter(case.label for case in bundle.graph.cases if case.source_dataset in sources)
        )
    output = {
        "schema_version": "filtered-rule-distribution-audit-v1",
        "knowledge_bundle_hash": bundle.content_hash(),
        "evidence_graph_version": bundle.graph.version,
        "manifest_hash": manifest_hash,
        "llm_calls": 0,
        "label_usage": "labels read only after features, matches and routes are frozen; metrics only",
        "splits": reports,
    }
    _write_json(args.output_dir / "distribution_audit.json", output)
    _write_report(args.output_dir / "README.md", output)
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
