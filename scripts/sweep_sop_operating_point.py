#!/usr/bin/env python3
"""在训练集 LOO 上扫描 SOP 操作点，并用 bootstrap 检验差异是否是噪声。

`ablate_feature_families.py` 已经给出一个不舒服的结论：全特征 SOP 的 LOO 准确率
（0.6335）的 Wilson 下界低于多数类先验（0.6211），即「按 SOP 判」和「一律报 L2」
在统计上分不开。既然如此，调阈值就不能按「哪个配置在 LOO 上最高」来挑——
161 条样本上 2-3 个点的差距完全在噪声范围内，按最高值挑等于把噪声当结论。

所以本脚本对每个配置同时给三样东西：

1. LOO 点估计：准确率、门限下的覆盖率与精度。
2. Bootstrap 区间：对 LOO 预测结果重采样 B 次，给出准确率与选择性精度的 2.5/97.5 分位。
3. `wins_vs_majority`：bootstrap 中该配置严格优于「一律报多数类」的比例。
   这个数才是判断配置值不值得上线的依据。低于 0.9 的配置不应被当成改进。

扫描维度：SOP 树深度、最小叶子、屏蔽的特征家族、以及保留答案所需的 Wilson 下界门限。
所有标签只来自 train split；测试集在本脚本中完全不出现。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca_framework.anomaly import fit_thresholds  # noqa: E402
from rca_framework.branches.base import wilson_lower_bound  # noqa: E402
from rca_framework.data import cases_by_manifest_split  # noqa: E402
from rca_framework.evidence_pack import build_packs  # noqa: E402
from rca_framework.features import dictionary_for, extract_features, fit_feature_model  # noqa: E402
from rca_framework.features.extractor import CaseFeatures  # noqa: E402
from rca_framework.sop.library import learn_sop  # noqa: E402

#: 屏蔽方案。键是报告里用的短名，值是被移除的 token 前缀。
MASKS: Dict[str, Tuple[str, ...]] = {
    "full": (),
    "no_imbalance": ("imbalance:",),
    "no_level": ("level:",),
    "no_imbalance_level": ("imbalance:", "level:"),
    "no_tx_level": ("level:L1:txpower_mean:", "level:L2:txpower_mean:"),
    "no_imbalance_tx_level": ("imbalance:", "level:L1:txpower_mean:", "level:L2:txpower_mean:"),
}


def mask_features(features: CaseFeatures, prefixes: Sequence[str]) -> CaseFeatures:
    if not prefixes:
        return features
    kept = tuple(
        token for token in features.tokens if not any(token.startswith(prefix) for prefix in prefixes)
    )
    return dataclasses.replace(features, tokens=kept)


@dataclasses.dataclass(frozen=True)
class LOORecord:
    truth: str
    verdict: str
    lower_bound: float
    support: int


def loo_records(
    features: Sequence[CaseFeatures],
    labels: Sequence[str],
    *,
    max_depth: int,
    min_leaf_size: int,
) -> Tuple[List[LOORecord], Counter]:
    records: List[LOORecord] = []
    roots: Counter = Counter()
    for index in range(len(features)):
        train_features = [item for position, item in enumerate(features) if position != index]
        train_labels = [item for position, item in enumerate(labels) if position != index]
        sop = learn_sop(
            train_features,
            train_labels,
            max_depth=max_depth,
            min_leaf_size=min_leaf_size,
            source="loo",
        )
        roots[sop.root.token or "<leaf>"] += 1
        prediction = sop.predict(features[index])
        records.append(
            LOORecord(
                truth=labels[index],
                verdict=prediction.verdict or "",
                lower_bound=prediction.confidence_lower_bound,
                support=prediction.support,
            )
        )
    return records, roots


def score(records: Sequence[LOORecord], gate: float, majority: str) -> Dict[str, float]:
    """门限内的精度必须和**同一子集**上的多数类基线比。

    高置信子集本身会富集多数类，所以拿子集精度去比全量先验永远赢，
    那个比较测的是「门限挑出了容易的 case」，不是「SOP 比拍多数类强」。
    `majority_on_kept` 就是同一批被保留的 case 上一律报多数类的正确率。
    """
    total = len(records)
    if not total:
        return {}
    correct = sum(1 for item in records if item.verdict and item.truth == item.verdict)
    kept = [item for item in records if item.verdict and item.lower_bound >= gate]
    kept_correct = sum(1 for item in kept if item.truth == item.verdict)
    return {
        "accuracy": correct / total,
        "coverage": len(kept) / total,
        "selective_precision": (kept_correct / len(kept)) if kept else 0.0,
        "manual_rate": 1.0 - len(kept) / total,
        "majority_accuracy": sum(1 for item in records if item.truth == majority) / total,
        "majority_on_kept": (
            sum(1 for item in kept if item.truth == majority) / len(kept) if kept else 0.0
        ),
    }


def bootstrap(
    records: Sequence[LOORecord],
    gate: float,
    majority: str,
    *,
    rounds: int,
    seed: int,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    total = len(records)
    accuracies: List[float] = []
    precisions: List[float] = []
    coverages: List[float] = []
    wins = 0
    precision_beats_majority = 0
    for _ in range(rounds):
        sample = [records[rng.randrange(total)] for _ in range(total)]
        stats = score(sample, gate, majority)
        accuracies.append(stats["accuracy"])
        coverages.append(stats["coverage"])
        precisions.append(stats["selective_precision"])
        if stats["accuracy"] > stats["majority_accuracy"]:
            wins += 1
        if stats["selective_precision"] > stats["majority_on_kept"]:
            precision_beats_majority += 1

    def interval(values: List[float]) -> List[float]:
        ordered = sorted(values)
        low = ordered[int(0.025 * len(ordered))]
        high = ordered[min(len(ordered) - 1, int(0.975 * len(ordered)))]
        return [round(low, 4), round(high, 4)]

    return {
        "rounds": rounds,
        "accuracy_ci95": interval(accuracies),
        "coverage_ci95": interval(coverages),
        "selective_precision_ci95": interval(precisions),
        "wins_vs_majority": round(wins / rounds, 4),
        "selective_precision_beats_majority": round(precision_beats_majority / rounds, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/rca_v2_l2fixed"))
    parser.add_argument("--feature-profile", default="v2")
    parser.add_argument("--masks", nargs="*", default=["full", "no_imbalance", "no_imbalance_level"])
    parser.add_argument("--depths", nargs="*", type=int, default=[2, 3, 4])
    parser.add_argument("--min-leaf-sizes", nargs="*", type=int, default=[5, 8, 12])
    parser.add_argument("--gates", nargs="*", type=float, default=[0.0, 0.4, 0.5, 0.55, 0.6])
    parser.add_argument("--bootstrap-rounds", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts/i1_sop_operating_points.json"))
    args = parser.parse_args()

    train_cases = cases_by_manifest_split(args.data_dir, "train")
    labels = [str(case["label"]) for case in train_cases]
    dictionary = dictionary_for(args.feature_profile)
    packs = build_packs(train_cases, source_dataset=str(args.data_dir))
    thresholds = fit_thresholds(train_cases)
    model = fit_feature_model(packs, dictionary=dictionary)
    base_features = [extract_features(pack, thresholds, model, dictionary=dictionary) for pack in packs]

    counts = Counter(labels)
    majority = max(counts, key=lambda label: counts[label])
    majority_rate = counts[majority] / len(labels)
    print(f"train={len(labels)} 多数类={majority} 先验={round(majority_rate, 4)}")
    print(f"bootstrap={args.bootstrap_rounds} 轮，seed={args.seed}\n")

    rows: List[Dict[str, Any]] = []
    for mask_name in args.masks:
        prefixes = MASKS[mask_name]
        masked = [mask_features(item, prefixes) for item in base_features]
        for depth in args.depths:
            for leaf in args.min_leaf_sizes:
                records, roots = loo_records(
                    masked, labels, max_depth=depth, min_leaf_size=leaf
                )
                for gate in args.gates:
                    stats = score(records, gate, majority)
                    boot = bootstrap(
                        records,
                        gate,
                        majority,
                        rounds=args.bootstrap_rounds,
                        seed=args.seed,
                    )
                    kept = [item for item in records if item.verdict and item.lower_bound >= gate]
                    kept_correct = sum(1 for item in kept if item.truth == item.verdict)
                    rows.append(
                        {
                            "mask": mask_name,
                            "dropped_prefixes": list(prefixes),
                            "max_depth": depth,
                            "min_leaf_size": leaf,
                            "gate": gate,
                            "accuracy": round(stats["accuracy"], 4),
                            "coverage": round(stats["coverage"], 4),
                            "manual_rate": round(stats["manual_rate"], 4),
                            "selective_precision": round(stats["selective_precision"], 4),
                            "majority_on_kept": round(stats["majority_on_kept"], 4),
                            "lift_over_majority_on_kept": round(
                                stats["selective_precision"] - stats["majority_on_kept"], 4
                            ),
                            "selective_precision_wilson_lower_bound": wilson_lower_bound(
                                kept_correct, len(kept)
                            ),
                            "kept": len(kept),
                            "bootstrap": boot,
                            "root_split_token": next(iter(roots), ""),
                        }
                    )

    # 报告排序：先看是否稳定优于多数类，再看覆盖率。
    rows.sort(
        key=lambda row: (
            -row["bootstrap"]["selective_precision_beats_majority"],
            -row["coverage"],
        )
    )

    print(
        f"{'mask':>22} {'d':>2} {'leaf':>4} {'gate':>5} {'cover':>6} {'prec':>6}"
        f" {'maj_kept':>8} {'lift':>6} {'beats_maj':>9} {'人工':>6}"
    )
    for row in rows[:24]:
        print(
            f"{row['mask']:>22} {row['max_depth']:>2} {row['min_leaf_size']:>4}"
            f" {row['gate']:>5.2f} {row['coverage']:>6.3f} {row['selective_precision']:>6.3f}"
            f" {row['majority_on_kept']:>8.3f} {row['lift_over_majority_on_kept']:>+6.3f}"
            f" {row['bootstrap']['selective_precision_beats_majority']:>9.3f}"
            f" {row['manual_rate']:>6.3f}"
        )

    report = {
        "schema_version": "sop-operating-points-v1",
        "data_dir": str(args.data_dir),
        "feature_dictionary_version": dictionary.version,
        "train_size": len(labels),
        "majority_label": majority,
        "majority_rate": round(majority_rate, 4),
        "protocol": (
            "留一法产出无偏预测，再对 LOO 预测做 bootstrap；"
            "标签只来自 train split，测试集不参与。"
        ),
        "selection_caveat": (
            "本表是在同一份 161 条 LOO 结果上扫出来的，直接取最大值会带来选择偏差。"
            "选配置时应优先看 selective_precision_beats_majority 是否稳定高，"
            "而不是看点估计差 1-2 个百分点。"
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
