"""T1 家族消融：枚举特征字典家族的所有子集，给出分辨率与可匹配性的权衡前沿。

背景：提高 signature 分辨率与保住 N5a（`sim = 100%`）桶是一对直接冲突的目标。
特征越细，混合标签 signature 越少，但完全匹配也越少。单看任何一个指标都会得到
误导性的结论，因此这里对每个子集同时报告两侧指标，再由人选定 v1。

逐家族 token 只算一次，之后按子集做并集，因此 1024 个子集可以在秒级跑完。
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze_signature_resolution import (  # noqa: E402
    EXACT_THRESHOLD,
    PARTIAL_THRESHOLD,
    idf_weights,
    signature_key,
    strip_label,
    weighted_jaccard,
)
from rca_framework.anomaly import fit_thresholds  # noqa: E402
from rca_framework.data import load_cases  # noqa: E402
from rca_framework.evidence_pack import EvidencePack  # noqa: E402
from rca_framework.features.dictionary import FULL_DICTIONARY  # noqa: E402
from rca_framework.features.extractor import (  # noqa: E402
    FAMILY_EXTRACTORS,
    fit_feature_model,
)
from rca_framework.types import ROOT_CAUSES  # noqa: E402


def family_tokens(
    packs: Sequence[EvidencePack],
    thresholds: Any,
    model: Any,
) -> List[Dict[str, Set[str]]]:
    return [
        {
            name: set(extractor(pack.telemetry, thresholds, model))
            for name, extractor in FAMILY_EXTRACTORS.items()
        }
        for pack in packs
    ]


def match_metrics(
    query_signatures: Sequence[Set[str]],
    query_labels: Sequence[str],
    index_signatures: Sequence[Set[str]],
    index_labels: Sequence[str],
    idf: Dict[str, float],
    *,
    skip_self: bool = False,
) -> Dict[str, Any]:
    """对一批 query 做 Top-N 匹配并按 N4 分支汇总多数投票准确率。

    `skip_self=True` 时 query 与 index 是同一批 case，跳过自身，即留一法。
    """
    branch_counts: Counter[str] = Counter()
    branch_correct: Counter[str] = Counter()
    overall_correct = 0
    for position, (label, signature) in enumerate(zip(query_labels, query_signatures)):
        scored = [
            (weighted_jaccard(signature, candidate, idf), index)
            for index, candidate in enumerate(index_signatures)
            if not (skip_self and index == position)
        ]
        best = max((score for score, _ in scored), default=0.0)
        tied = [index_labels[index] for score, index in scored if score == best and best > 0.0]
        branch = "N5a" if best >= EXACT_THRESHOLD else "N5b" if best >= PARTIAL_THRESHOLD else "N5c"
        branch_counts[branch] += 1
        if not tied:
            continue
        vote = Counter(tied)
        top = max(vote.values())
        majority = min((lab for lab in vote if vote[lab] == top), key=ROOT_CAUSES.index)
        branch_correct[branch] += int(majority == label)
        overall_correct += int(majority == label)

    result: Dict[str, Any] = {
        "overall_majority_accuracy": round(overall_correct / len(query_labels), 6) if query_labels else None,
    }
    for branch in ("N5a", "N5b", "N5c"):
        count = branch_counts[branch]
        result[f"{branch.lower()}_count"] = count
        result[f"{branch.lower()}_majority_correct"] = branch_correct[branch]
        result[f"{branch.lower()}_majority_accuracy"] = round(branch_correct[branch] / count, 6) if count else None
    return result


def evaluate_subset(
    families: Tuple[str, ...],
    train_tokens: List[Dict[str, Set[str]]],
    train_labels: List[str],
    test_tokens: List[Dict[str, Set[str]]],
    test_labels: List[str],
) -> Dict[str, Any]:
    train_sig = [set().union(*(row[name] for name in families)) if families else set() for row in train_tokens]
    test_sig = [set().union(*(row[name] for name in families)) if families else set() for row in test_tokens]
    idf = idf_weights(train_sig)

    groups: Dict[str, List[str]] = defaultdict(list)
    for label, signature in zip(train_labels, train_sig):
        groups[signature_key(signature)].append(label)
    mixed_cases = sum(len(labels) for labels in groups.values() if len(set(labels)) > 1)

    loo = match_metrics(train_sig, train_labels, train_sig, train_labels, idf, skip_self=True)
    held_out = match_metrics(test_sig, test_labels, train_sig, train_labels, idf)

    return {
        "families": list(families),
        "family_count": len(families),
        "distinct_tokens": len(idf),
        "train_groups": len(groups),
        "mixed_label_case_ratio": round(mixed_cases / len(train_labels), 6),
        "train_loo": loo,
        "test": held_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/organized_rca_v2_stratified_60_40_seed42"))
    parser.add_argument("--train-size", type=int, default=126)
    parser.add_argument("--min-n5a", type=int, default=20, help="报告时要求 N5a 桶至少有多少 case 才算可用")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cases = load_cases(args.data_dir)
    train_cases, test_cases = cases[: args.train_size], cases[args.train_size :]
    thresholds = fit_thresholds(train_cases)
    train_packs = [EvidencePack.from_case(case) for case in train_cases]
    test_packs = [EvidencePack.from_case(case) for case in test_cases]
    model = fit_feature_model(train_packs)

    train_tokens = family_tokens(train_packs, thresholds, model)
    test_tokens = family_tokens(test_packs, thresholds, model)
    train_labels = [str(case["label"]) for case in train_cases]
    test_labels = [str(case["label"]) for case in test_cases]

    names = list(FULL_DICTIONARY.family_names())
    results: List[Dict[str, Any]] = []
    for size in range(1, len(names) + 1):
        for subset in itertools.combinations(names, size):
            results.append(
                evaluate_subset(subset, train_tokens, train_labels, test_tokens, test_labels)
            )

    usable = [row for row in results if row["train_loo"]["n5a_count"] >= args.min_n5a]
    print(f"subsets evaluated : {len(results)}")
    print(f"subsets with train-LOO N5a >= {args.min_n5a}: {len(usable)}\n")
    print("选型只看 train_loo 列；test 列一并打印是为了事后核对，不参与选型。\n")

    def show(title: str, rows: List[Dict[str, Any]], key, limit: int = 12) -> None:
        print(f"--- {title} ---")
        print(
            f"{'mixed%':>7} | {'N5a':>4} {'N5a maj':>8} {'all126':>7} | "
            f"{'N5a':>4} {'N5a maj':>8} {'all85':>7} | families"
        )
        print(f"{'':>7} | {'--- train LOO ---':^22} | {'--- held-out test ---':^22} |")
        for row in sorted(rows, key=key)[:limit]:
            loo, test = row["train_loo"], row["test"]

            def pct(value: Any) -> str:
                return f"{value:.2%}" if value is not None else "-"

            print(
                f"{row['mixed_label_case_ratio']:>7.2%} | "
                f"{loo['n5a_count']:>4} {pct(loo['n5a_majority_accuracy']):>8} "
                f"{pct(loo['overall_majority_accuracy']):>7} | "
                f"{test['n5a_count']:>4} {pct(test['n5a_majority_accuracy']):>8} "
                f"{pct(test['overall_majority_accuracy']):>7} | {'+'.join(row['families'])}"
            )
        print()

    show(
        "按 train LOO 全集准确率选型（N5a 桶可用）",
        usable,
        lambda row: -(row["train_loo"]["overall_majority_accuracy"] or 0.0),
    )
    show(
        "按 train LOO 的 N5a 桶内多数投票准确率选型（N5a 桶可用）",
        usable,
        lambda row: -(row["train_loo"]["n5a_majority_accuracy"] or 0.0),
    )
    show(
        "最低混合标签覆盖率（N5a 桶可用）",
        usable,
        lambda row: (row["mixed_label_case_ratio"], -row["train_loo"]["n5a_count"]),
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"min_n5a": args.min_n5a, "results": results}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
