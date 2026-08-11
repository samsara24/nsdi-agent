#!/usr/bin/env python3
"""嵌套验证：把「挑配置」这一步也算进代价里。

`sweep_sop_operating_point.py` 在 270 个配置上扫出一个 lift +9.7pp 的操作点。
问题是这个数字和它被挑出来的过程是同一批数据算的——在 161 条样本上试 270 次，
最大值本身就带着可观的向上偏差。直接把它当成上线后的预期收益是错的。

本脚本用嵌套协议给出**包含配置选择过程**的无偏估计：

- 外层：分层 K 折。每折的 held-out 完全不参与任何选择。
- 内层：只在该折的训练部分做 LOO，在候选配置里按 lift 挑一个。
- 评估：用挑中的配置在训练部分重学 SOP，预测 held-out。

汇总所有折的 held-out 预测后，报告的 lift 才是「这套自动选配置的流程」的真实收益。
同时输出每折选中的配置，若各折选择互不一致，说明这个操作点本身不稳定，
不管点估计多高都不该写进默认策略。

`majority` 配置（永远报多数类、不弃答）作为参照放进候选集：
如果内层经常选它，说明 SOP 在这份数据上没有立足之地。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca_framework.anomaly import fit_thresholds  # noqa: E402
from rca_framework.branches.base import wilson_lower_bound  # noqa: E402
from rca_framework.data import cases_by_manifest_split  # noqa: E402
from rca_framework.evidence_pack import build_packs  # noqa: E402
from rca_framework.features import dictionary_for, extract_features, fit_feature_model  # noqa: E402
from rca_framework.features.extractor import CaseFeatures  # noqa: E402
from rca_framework.sop.library import learn_sop  # noqa: E402
from scripts.sweep_sop_operating_point import MASKS, mask_features  # noqa: E402


@dataclasses.dataclass(frozen=True)
class Config:
    mask: str
    max_depth: int
    min_leaf_size: int
    gate: float

    @property
    def name(self) -> str:
        if self.mask == "majority":
            return "majority"
        return f"{self.mask}/d{self.max_depth}/leaf{self.min_leaf_size}/gate{self.gate:g}"


#: 候选配置。刻意保持小而有理由：扫描里 lift 为正的都是「细树 + 低门限」，
#: 再加上两个粗配置和多数类参照，用来暴露过细配置的不稳定性。
CANDIDATES: Tuple[Config, ...] = (
    Config("majority", 0, 0, 0.0),
    Config("full", 2, 5, 0.4),
    Config("full", 3, 5, 0.4),
    Config("full", 4, 5, 0.4),
    Config("full", 3, 12, 0.4),
    Config("no_imbalance", 3, 5, 0.4),
    Config("no_imbalance", 4, 5, 0.4),
    Config("no_imbalance", 4, 5, 0.5),
    Config("no_imbalance_level", 3, 5, 0.4),
)


@dataclasses.dataclass(frozen=True)
class Prediction:
    truth: str
    verdict: str
    kept: bool


def evaluate(predictions: Sequence[Prediction], majority: str) -> Dict[str, Any]:
    total = len(predictions)
    kept = [item for item in predictions if item.kept and item.verdict]
    kept_correct = sum(1 for item in kept if item.truth == item.verdict)
    majority_on_kept = sum(1 for item in kept if item.truth == majority)
    answered_correct = sum(1 for item in predictions if item.verdict and item.truth == item.verdict)
    return {
        "cases": total,
        "coverage": round(len(kept) / total, 4) if total else 0.0,
        "manual_rate": round(1.0 - len(kept) / total, 4) if total else 0.0,
        "accuracy_all": round(answered_correct / total, 4) if total else 0.0,
        "selective_precision": round(kept_correct / len(kept), 4) if kept else 0.0,
        "selective_precision_wilson_lower_bound": wilson_lower_bound(kept_correct, len(kept)),
        "majority_on_kept": round(majority_on_kept / len(kept), 4) if kept else 0.0,
        "lift_over_majority_on_kept": round(
            (kept_correct - majority_on_kept) / len(kept), 4
        ) if kept else 0.0,
        "kept": len(kept),
    }


def predict_with_config(
    config: Config,
    train_features: Sequence[CaseFeatures],
    train_labels: Sequence[str],
    eval_features: Sequence[CaseFeatures],
    eval_labels: Sequence[str],
    majority: str,
) -> List[Prediction]:
    if config.mask == "majority":
        return [Prediction(truth=truth, verdict=majority, kept=True) for truth in eval_labels]
    prefixes = MASKS[config.mask]
    masked_train = [mask_features(item, prefixes) for item in train_features]
    sop = learn_sop(
        masked_train,
        list(train_labels),
        max_depth=config.max_depth,
        min_leaf_size=config.min_leaf_size,
        source="nested",
    )
    out: List[Prediction] = []
    for features, truth in zip(eval_features, eval_labels):
        prediction = sop.predict(mask_features(features, prefixes))
        out.append(
            Prediction(
                truth=truth,
                verdict=prediction.verdict or "",
                kept=bool(prediction.verdict) and prediction.confidence_lower_bound >= config.gate,
            )
        )
    return out


def inner_loo(
    config: Config,
    features: Sequence[CaseFeatures],
    labels: Sequence[str],
    majority: str,
) -> Dict[str, Any]:
    predictions: List[Prediction] = []
    for index in range(len(features)):
        train_features = [item for position, item in enumerate(features) if position != index]
        train_labels = [item for position, item in enumerate(labels) if position != index]
        predictions.extend(
            predict_with_config(
                config,
                train_features,
                train_labels,
                [features[index]],
                [labels[index]],
                majority,
            )
        )
    return evaluate(predictions, majority)


def stratified_folds(labels: Sequence[str], folds: int, seed: int) -> List[List[int]]:
    buckets: Dict[str, List[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        buckets[label].append(index)
    rng = random.Random(seed)
    assignment: List[List[int]] = [[] for _ in range(folds)]
    for label in sorted(buckets):
        indices = buckets[label][:]
        rng.shuffle(indices)
        for position, index in enumerate(indices):
            assignment[position % folds].append(index)
    return assignment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/rca_v2_l2fixed"))
    parser.add_argument("--feature-profile", default="v2")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-inner-coverage",
        type=float,
        default=0.5,
        help="内层选配置时要求的最低覆盖率，避免选出一个只答几条的配置。",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/i1_nested_validation.json"))
    args = parser.parse_args()

    train_cases = cases_by_manifest_split(args.data_dir, "train")
    labels = [str(case["label"]) for case in train_cases]
    dictionary = dictionary_for(args.feature_profile)
    packs = build_packs(train_cases, source_dataset=str(args.data_dir))
    thresholds = fit_thresholds(train_cases)
    model = fit_feature_model(packs, dictionary=dictionary)
    features = [extract_features(pack, thresholds, model, dictionary=dictionary) for pack in packs]

    counts = Counter(labels)
    majority = max(counts, key=lambda label: counts[label])
    print(f"train={len(labels)} 多数类={majority} 先验={round(counts[majority]/len(labels),4)}")
    print(f"外层 {args.folds} 折分层，内层 LOO 选配置，候选 {len(CANDIDATES)} 个\n")

    folds = stratified_folds(labels, args.folds, args.seed)
    held_out: List[Prediction] = []
    fold_reports: List[Dict[str, Any]] = []
    chosen_counter: Counter = Counter()

    for fold_index, fold in enumerate(folds):
        fold_set = set(fold)
        inner_indices = [index for index in range(len(labels)) if index not in fold_set]
        inner_features = [features[index] for index in inner_indices]
        inner_labels = [labels[index] for index in inner_indices]

        scores: List[Tuple[Config, Dict[str, Any]]] = []
        for config in CANDIDATES:
            stats = inner_loo(config, inner_features, inner_labels, majority)
            scores.append((config, stats))

        eligible = [
            item for item in scores if item[1]["coverage"] >= args.min_inner_coverage
        ] or scores
        best_config, best_stats = max(
            eligible, key=lambda item: (item[1]["lift_over_majority_on_kept"], item[1]["coverage"])
        )
        chosen_counter[best_config.name] += 1

        predictions = predict_with_config(
            best_config,
            inner_features,
            inner_labels,
            [features[index] for index in fold],
            [labels[index] for index in fold],
            majority,
        )
        held_out.extend(predictions)
        outer = evaluate(predictions, majority)
        fold_reports.append(
            {
                "fold": fold_index,
                "held_out_size": len(fold),
                "chosen_config": best_config.name,
                "inner_lift": best_stats["lift_over_majority_on_kept"],
                "inner_selective_precision": best_stats["selective_precision"],
                "inner_coverage": best_stats["coverage"],
                "held_out": outer,
                "inner_ranking": [
                    {
                        "config": config.name,
                        "lift": stats["lift_over_majority_on_kept"],
                        "coverage": stats["coverage"],
                        "selective_precision": stats["selective_precision"],
                    }
                    for config, stats in sorted(
                        scores, key=lambda item: -item[1]["lift_over_majority_on_kept"]
                    )
                ],
            }
        )
        print(
            f"fold {fold_index}: 内层选中 {best_config.name}"
            f" (内层 lift {best_stats['lift_over_majority_on_kept']:+.4f},"
            f" cover {best_stats['coverage']:.3f})"
            f" → held-out cover {outer['coverage']:.3f}"
            f" prec {outer['selective_precision']:.3f}"
            f" maj {outer['majority_on_kept']:.3f}"
            f" lift {outer['lift_over_majority_on_kept']:+.4f}"
        )

    pooled = evaluate(held_out, majority)
    print()
    print("== 汇总 held-out（包含配置选择代价的无偏估计）==")
    print(
        f"覆盖 {pooled['coverage']:.4f} | 人工 {pooled['manual_rate']:.4f}"
        f" | 选择性精度 {pooled['selective_precision']:.4f}"
        f" (95%下界 {pooled['selective_precision_wilson_lower_bound']:.4f})"
        f" | 同子集多数类 {pooled['majority_on_kept']:.4f}"
        f" | lift {pooled['lift_over_majority_on_kept']:+.4f}"
    )
    print(f"各折选中的配置: {dict(chosen_counter)}")

    # 同一协议下，多数类参照的 held-out 表现（不弃答）。
    majority_only = [Prediction(truth=item.truth, verdict=majority, kept=True) for item in held_out]
    majority_stats = evaluate(majority_only, majority)
    print(
        f"多数类参照: 覆盖 {majority_stats['coverage']:.4f}"
        f" 精度 {majority_stats['selective_precision']:.4f}"
    )

    report = {
        "schema_version": "nested-validation-v1",
        "data_dir": str(args.data_dir),
        "feature_dictionary_version": dictionary.version,
        "train_size": len(labels),
        "majority_label": majority,
        "folds": args.folds,
        "seed": args.seed,
        "min_inner_coverage": args.min_inner_coverage,
        "candidates": [config.name for config in CANDIDATES],
        "protocol": (
            "外层分层 K 折；内层只在训练部分做 LOO 选配置；held-out 不参与选择。"
            "汇总的 lift 已包含配置选择带来的偏差。"
        ),
        "pooled_held_out": pooled,
        "majority_reference": majority_stats,
        "chosen_configs": dict(chosen_counter),
        "folds_detail": fold_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
