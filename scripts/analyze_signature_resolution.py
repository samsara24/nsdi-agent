"""T1 验收脚本：量化比较不同特征集合的 signature 分辨率与 N4 分流表现。

用法::

    python scripts/analyze_signature_resolution.py --feature-set legacy
    python scripts/analyze_signature_resolution.py --feature-set v1 --output artifacts/t1_v1.json

`legacy` 口径直接取 `extract_evidence` 产出的 `anomaly_id` 集合，用于复现
Progress.md 第 6 节已记录的基线数字；其余口径来自 `features/extractor.py`。
两条口径共用同一份切分、同一份阈值拟合和同一个 IDF 加权 Jaccard 检索，
因此差异只来自特征集合本身。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Set, Tuple

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca_framework.anomaly import ThresholdModel, extract_evidence, fit_thresholds
from rca_framework.data import load_cases
from rca_framework.evidence_pack import EvidencePack
from rca_framework.types import ROOT_CAUSES


EXACT_THRESHOLD = 1.0
PARTIAL_THRESHOLD = 0.7

FeatureFn = Callable[[EvidencePack, ThresholdModel], Set[str]]


def legacy_features(pack: EvidencePack, thresholds: ThresholdModel) -> Set[str]:
    return set(extract_evidence(pack.telemetry, thresholds).anomaly_ids)


def build_feature_fn(
    name: str,
    train_packs: Sequence[EvidencePack],
    thresholds: ThresholdModel,
) -> Tuple[FeatureFn, Dict[str, Any]]:
    """返回抽取函数和它的版本元数据。`FeatureModel` 只在训练集上拟合。"""
    if name == "legacy":
        return legacy_features, {"feature_set": "legacy", "source": "anomaly.extract_evidence"}

    from rca_framework.features.dictionary import dictionary_for
    from rca_framework.features.extractor import extract_feature_tokens, fit_feature_model

    dictionary = dictionary_for(name)
    model = fit_feature_model(train_packs, dictionary=dictionary)

    def fn(pack: EvidencePack, threshold_model: ThresholdModel) -> Set[str]:
        return set(extract_feature_tokens(pack, threshold_model, model, dictionary=dictionary))

    metadata = {
        "feature_set": name,
        "dictionary_version": dictionary.version,
        "dictionary_hash": dictionary.content_hash(),
        "families": list(dictionary.family_names()),
        "feature_model": model.to_dict(),
    }
    return fn, metadata


def idf_weights(signatures: Sequence[Set[str]]) -> Dict[str, float]:
    """与 `graph.AnomalyKnowledgeGraph.fit` 相同的 IDF 定义，保证两条口径可比。"""
    document_count = len(signatures)
    counts: Counter[str] = Counter()
    for signature in signatures:
        counts.update(signature)
    return {
        token: round(math.log((document_count + 1) / (count + 1)) + 1.0, 8)
        for token, count in sorted(counts.items())
    }


def weighted_jaccard(query: Set[str], candidate: Set[str], idf: Dict[str, float]) -> float:
    union = query | candidate
    if not union:
        # 两个空 signature 不算相似：否则所有零证据 case 会互相 100% 命中。
        return 0.0
    overlap = query & candidate
    numerator = sum(idf.get(token, 1.0) for token in sorted(overlap))
    denominator = sum(idf.get(token, 1.0) for token in sorted(union))
    return round(numerator / denominator, 8) if denominator else 0.0


def signature_key(signature: Set[str]) -> str:
    return "|".join(sorted(signature))


def train_signature_report(rows: Sequence[Tuple[str, str, Set[str]]]) -> Dict[str, Any]:
    groups: Dict[str, List[str]] = defaultdict(list)
    for _case_id, label, signature in rows:
        groups[signature_key(signature)].append(label)

    mixed_groups = {key: labels for key, labels in groups.items() if len(set(labels)) > 1}
    mixed_case_count = sum(len(labels) for labels in mixed_groups.values())
    empty_labels = groups.get("", [])
    singleton_groups = {key: labels for key, labels in groups.items() if len(labels) == 1}
    return {
        "train_case_count": len(rows),
        "distinct_signature_groups": len(groups),
        "mixed_label_group_count": len(mixed_groups),
        "mixed_label_case_count": mixed_case_count,
        "mixed_label_case_ratio": round(mixed_case_count / len(rows), 6) if rows else None,
        "pure_group_count": len(groups) - len(mixed_groups),
        "singleton_group_count": len(singleton_groups),
        "singleton_case_ratio": round(len(singleton_groups) / len(rows), 6) if rows else None,
        "empty_signature_case_count": len(empty_labels),
        "empty_signature_label_distribution": dict(Counter(empty_labels)),
        "mean_signature_size": round(sum(len(sig) for _, _, sig in rows) / len(rows), 6) if rows else None,
        "largest_mixed_groups": [
            {
                "size": len(labels),
                "labels": dict(Counter(labels)),
                "signature": key.split("|") if key else [],
            }
            for key, labels in sorted(mixed_groups.items(), key=lambda item: -len(item[1]))[:5]
        ],
    }


def route_report(
    train_rows: Sequence[Tuple[str, str, Set[str]]],
    test_rows: Sequence[Tuple[str, str, Set[str]]],
    idf: Dict[str, float],
) -> Dict[str, Any]:
    buckets: Dict[str, List[Dict[str, Any]]] = {"N5a": [], "N5b": [], "N5c": []}
    for case_id, label, signature in test_rows:
        scored = [
            (weighted_jaccard(signature, train_signature, idf), train_case_id, train_label)
            for train_case_id, train_label, train_signature in train_rows
        ]
        best = max((score for score, _, _ in scored), default=0.0)
        tied = [(cid, lab) for score, cid, lab in scored if score == best and best > 0.0]
        branch = "N5a" if best >= EXACT_THRESHOLD else "N5b" if best >= PARTIAL_THRESHOLD else "N5c"
        top1 = sorted(tied)[0][1] if tied else None
        vote = Counter(lab for _, lab in tied)
        majority = min(
            (lab for lab in vote if vote[lab] == max(vote.values())),
            key=lambda lab: ROOT_CAUSES.index(lab),
        ) if vote else None
        buckets[branch].append({
            "case_id": case_id,
            "actual_label": label,
            "similarity": best,
            "tie_count": len(tied),
            "top1_label": top1,
            "majority_label": majority,
            "oracle_hit": label in {lab for _, lab in tied},
            "tie_label_distribution": dict(vote),
        })

    def bucket_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(rows)
        if not total:
            return {"case_count": 0}
        top1 = sum(row["top1_label"] == row["actual_label"] for row in rows)
        majority = sum(row["majority_label"] == row["actual_label"] for row in rows)
        oracle = sum(row["oracle_hit"] for row in rows)
        pure = sum(len(row["tie_label_distribution"]) == 1 for row in rows)
        return {
            "case_count": total,
            "top1_correct": top1,
            "top1_accuracy": round(top1 / total, 6),
            "majority_correct": majority,
            "majority_accuracy": round(majority / total, 6),
            "oracle_correct": oracle,
            "oracle_accuracy": round(oracle / total, 6),
            "pure_tie_bucket_cases": pure,
            "pure_tie_bucket_ratio": round(pure / total, 6),
            "label_distribution": dict(Counter(row["actual_label"] for row in rows)),
            "per_label_majority_correct": {
                label: sum(
                    row["majority_label"] == row["actual_label"]
                    for row in rows
                    if row["actual_label"] == label
                )
                for label in ROOT_CAUSES
            },
        }

    return {
        "thresholds": {"exact": EXACT_THRESHOLD, "partial": PARTIAL_THRESHOLD},
        "distribution": {name: len(rows) for name, rows in buckets.items()},
        "N5a": bucket_metrics(buckets["N5a"]),
        "N5b": bucket_metrics(buckets["N5b"]),
        "N5c": bucket_metrics(buckets["N5c"]),
        "zero_signature_test_cases": sum(1 for _, _, sig in test_rows if not sig),
        "cases": {name: rows for name, rows in buckets.items()},
    }


def analyze(
    data_dir: Path,
    train_size: int,
    feature_set: str,
) -> Dict[str, Any]:
    cases = load_cases(data_dir)
    train_cases, test_cases = cases[:train_size], cases[train_size:]
    thresholds = fit_thresholds(train_cases)
    train_packs = [EvidencePack.from_case(case, source_dataset=str(data_dir)) for case in train_cases]
    test_packs = [EvidencePack.from_case(case, source_dataset=str(data_dir)) for case in test_cases]
    feature_fn, feature_metadata = build_feature_fn(feature_set, train_packs, thresholds)

    train_rows = [
        (pack.case_id, str(case.get("label")), feature_fn(pack, thresholds))
        for case, pack in zip(train_cases, train_packs)
    ]
    test_rows = [
        (pack.case_id, str(case.get("label")), feature_fn(pack, thresholds))
        for case, pack in zip(test_cases, test_packs)
    ]
    idf = idf_weights([signature for _, _, signature in train_rows])

    report = {
        "feature_set": feature_set,
        "data_dir": str(data_dir),
        "train_size": len(train_rows),
        "test_size": len(test_rows),
        "distinct_feature_tokens": len(idf),
        "train_label_distribution": dict(Counter(label for _, label, _ in train_rows)),
        "test_label_distribution": dict(Counter(label for _, label, _ in test_rows)),
        "signature_resolution": train_signature_report(train_rows),
        "routing": route_report(train_rows, test_rows, idf),
        "feature_metadata": feature_metadata,
    }
    return report


def strip_label(case: Dict[str, Any]) -> Dict[str, Any]:
    target = dict(case)
    target.pop("label", None)
    return target


def summarize(report: Dict[str, Any]) -> str:
    resolution = report["signature_resolution"]
    routing = report["routing"]
    lines = [
        f"feature_set              : {report['feature_set']}",
        f"distinct feature tokens  : {report['distinct_feature_tokens']}",
        f"train signature groups   : {resolution['distinct_signature_groups']} / {resolution['train_case_count']}",
        f"mixed-label groups       : {resolution['mixed_label_group_count']} groups covering "
        f"{resolution['mixed_label_case_count']} cases ({resolution['mixed_label_case_ratio']:.2%})",
        f"singleton groups         : {resolution['singleton_group_count']} "
        f"({resolution['singleton_case_ratio']:.2%} of train cases)",
        f"empty signature cases    : {resolution['empty_signature_case_count']} "
        f"{resolution['empty_signature_label_distribution']}",
        f"N4 distribution          : {routing['distribution']}",
    ]
    for branch in ("N5a", "N5b", "N5c"):
        bucket = routing[branch]
        if not bucket["case_count"]:
            lines.append(f"{branch}                     : 0 case")
            continue
        lines.append(
            f"{branch} n={bucket['case_count']:<3} top1={bucket['top1_correct']}/{bucket['case_count']}"
            f"={bucket['top1_accuracy']:.2%} majority={bucket['majority_correct']}/{bucket['case_count']}"
            f"={bucket['majority_accuracy']:.2%} oracle={bucket['oracle_accuracy']:.2%}"
            f" pure_tie={bucket['pure_tie_bucket_ratio']:.2%}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/organized_rca_v2_stratified_60_40_seed42"))
    parser.add_argument("--train-size", type=int, default=126)
    parser.add_argument("--feature-set", default="legacy")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-cases", action="store_true", help="不在 JSON 中写入逐 case 明细")
    args = parser.parse_args()

    report = analyze(args.data_dir, args.train_size, args.feature_set)
    print(summarize(report))
    if args.output:
        payload = json.loads(json.dumps(report))
        if args.no_cases:
            payload["routing"].pop("cases", None)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
