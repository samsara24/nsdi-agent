"""T4 阈值标定：给 N4 分流阈值画 coverage-accuracy 曲线。

背景见 `Validation.md` V1。画板定稿的 `sim = 100%` / `70%` 是在 legacy `anomaly_id`
特征空间上定的；换到特征字典 v1 之后完全匹配从 46/85 掉到 21/85，阈值需要重定。
这个脚本不替人做决定，它产出的是决策所需的曲线：给定一个相似度阈值，
有多少比例的 case 会落在阈值之上（coverage），落在阈值之上的那批准确率是多少
（precision_at_coverage）。

标定必须只用训练集留一法。测试集曲线一并打印，但只用于事后核对，不参与选阈值——
否则等于用测试集调超参。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca_framework.anomaly import fit_thresholds  # noqa: E402
from rca_framework.data import load_cases  # noqa: E402
from rca_framework.evidence_graph import EvidenceGraph, MatchResult, match_many  # noqa: E402
from rca_framework.evidence_pack import build_packs, labels_of  # noqa: E402
from rca_framework.features.dictionary import dictionary_for  # noqa: E402
from rca_framework.features.extractor import extract_features, fit_feature_model  # noqa: E402
from rca_framework.types import ROOT_CAUSES  # noqa: E402


#: 候选阈值网格。0.999 与 1.0 分开列，用来看「严格完全匹配」和「几乎完全匹配」差多少。
THRESHOLD_GRID = (
    1.0, 0.999, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.4, 0.3, 0.2, 0.0,
)


def majority_label(result: MatchResult) -> str | None:
    labels = result.tie_labels
    if not labels:
        return None
    vote = Counter(labels)
    top = max(vote.values())
    return min((label for label in vote if vote[label] == top), key=ROOT_CAUSES.index)


def curve(results: Sequence[MatchResult], actual: Sequence[str]) -> List[Dict[str, Any]]:
    """对每个候选阈值，报告阈值之上的覆盖率与准确率。"""
    total = len(results)
    rows: List[Dict[str, Any]] = []
    for threshold in THRESHOLD_GRID:
        selected = [
            (result, label)
            for result, label in zip(results, actual)
            if result.max_similarity >= threshold
        ]
        correct = sum(majority_label(result) == label for result, label in selected)
        pure = sum(result.is_label_pure for result, _ in selected)
        conflicted = sum(result.has_conflict for result, _ in selected)
        rows.append({
            "threshold": threshold,
            "covered": len(selected),
            "coverage": round(len(selected) / total, 6) if total else 0.0,
            "correct": correct,
            "precision_at_coverage": round(correct / len(selected), 6) if selected else None,
            "pure_tie_ratio": round(pure / len(selected), 6) if selected else None,
            "conflicted_cases": conflicted,
            "fiber_covered": sum(1 for _, label in selected if label == "fiber"),
            "fiber_correct": sum(
                1 for result, label in selected if label == "fiber" and majority_label(result) == "fiber"
            ),
        })
    return rows


def build(data_dir: Path, train_size: int, feature_set: str) -> Dict[str, Any]:
    cases = load_cases(data_dir)
    train_cases, test_cases = cases[:train_size], cases[train_size:]
    thresholds = fit_thresholds(train_cases)
    dictionary = dictionary_for(feature_set)

    train_packs = build_packs(train_cases, source_dataset=str(data_dir))
    test_packs = build_packs(test_cases, source_dataset=str(data_dir))
    model = fit_feature_model(train_packs, dictionary=dictionary)

    train_features = [extract_features(pack, thresholds, model, dictionary=dictionary) for pack in train_packs]
    test_features = [extract_features(pack, thresholds, model, dictionary=dictionary) for pack in test_packs]

    graph = EvidenceGraph.build(
        train_features,
        labels_of(train_cases),
        feature_model=model,
        dictionary=dictionary,
        source_dataset=str(data_dir),
    )

    loo = match_many(graph, train_features, top_k=0, leave_one_out=True)
    held_out = match_many(graph, test_features, top_k=0)

    return {
        "feature_set": feature_set,
        "data_dir": str(data_dir),
        "graph_version": graph.version,
        "dictionary_version": graph.dictionary_version,
        "dictionary_hash": graph.dictionary_hash,
        "purity": graph.purity_report(),
        "train_loo_curve": curve(loo, labels_of(train_cases)),
        "held_out_curve": curve(held_out, labels_of(test_cases)),
        "train_prior": round(
            max(graph.label_distribution().values()) / len(train_cases), 6
        ),
    }


def show(title: str, rows: Sequence[Dict[str, Any]], baseline: float) -> None:
    print(f"--- {title} ---")
    print(f"{'阈值':>6} {'覆盖':>6} {'覆盖率':>8} {'准确':>6} {'阈上准确率':>10} {'桶纯净':>8} {'冲突':>5} {'fiber':>8}")
    for row in rows:
        precision = row["precision_at_coverage"]
        marker = ""
        if precision is not None and row["coverage"] >= 0.15:
            marker = " <-" if precision >= baseline + 0.10 else ""
        print(
            f"{row['threshold']:>6.3f} {row['covered']:>6} {row['coverage']:>8.2%} "
            f"{row['correct']:>6} "
            f"{(f'{precision:.2%}' if precision is not None else '-'):>10} "
            f"{(f'{row['pure_tie_ratio']:.2%}' if row['pure_tie_ratio'] is not None else '-'):>8} "
            f"{row['conflicted_cases']:>5} "
            f"{row['fiber_correct']}/{row['fiber_covered']:<6}{marker}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/organized_rca_v2_stratified_60_40_seed42"))
    parser.add_argument("--train-size", type=int, default=126)
    parser.add_argument("--feature-set", default="v1")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = build(args.data_dir, args.train_size, args.feature_set)
    print(f"feature_set   : {report['feature_set']}")
    print(f"graph_version : {report['graph_version']}")
    print(f"train prior   : {report['train_prior']:.2%}（阈上准确率必须显著高于它才有意义）")
    print(f"purity        : {report['purity']}\n")

    show("训练集留一法（用于选阈值）", report["train_loo_curve"], report["train_prior"])
    show("留出测试集（仅事后核对）", report["held_out_curve"], report["train_prior"])

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
