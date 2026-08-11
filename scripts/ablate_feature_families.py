#!/usr/bin/env python3
"""特征家族消融：用训练集 LOO 判断每个 token 家族是带来泛化，还是只带来拟合。

为什么需要这个脚本：`mine_knowledge_candidates.py` 报的 precision 是**同一批数据**
上的拟合值，一个家族哪怕纯粹是噪声，只要取值够细就能把训练集切干净。
要判断它是否值得写进知识层，必须看留一法下的表现——每次都把当前 case 排除后重学 SOP。

三个输出维度分别对应项目的三个优化目标：

- `loo_accuracy`：整体准确率。
- `selective_coverage` / `selective_precision`：在 Wilson 下界门限上保留多少条、保留部分准不准，
  直接决定人工干预比例。
- `root_split_tokens`：LOO 过程中被选为根分裂的 token 及次数。
  一个家族如果被屏蔽后根分裂换成了另一个 token 而指标不变，
  说明这两个 token 承载同一份信息，知识层只该保留物理上更站得住的那个。

`--drop-prefixes` 接受形如 `level:L1:txpower_mean:` 的前缀，命中前缀的 token 在
学习和预测时都被移除，等价于「这个家族不存在」。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
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

#: 每个待消融的家族给一个短名，便于在报告里引用。
ABLATIONS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("full", ()),
    ("no_tx_level", ("level:L1:txpower_mean:", "level:L2:txpower_mean:")),
    ("no_rx_level", ("level:L1:rxpower_mean:", "level:L2:rxpower_mean:")),
    ("no_snr_level", ("level:L1:media_snr_min:", "level:L2:media_snr_min:")),
    ("no_all_level", ("level:",)),
    ("no_drop", ("drop:",)),
    ("no_status", ("status:",)),
    ("no_lane", ("lane:",)),
    ("no_serdes_telemetry", ("serdes:", "telemetry:")),
    ("no_imbalance", ("imbalance:",)),
)


def mask(features: CaseFeatures, prefixes: Sequence[str]) -> CaseFeatures:
    if not prefixes:
        return features
    kept = tuple(
        token for token in features.tokens if not any(token.startswith(prefix) for prefix in prefixes)
    )
    return dataclasses.replace(features, tokens=kept)


def leave_one_out(
    features: Sequence[CaseFeatures],
    labels: Sequence[str],
    *,
    max_depth: int,
    min_leaf_size: int,
    lower_bound_gate: float,
) -> Dict[str, Any]:
    predictions: List[Tuple[str, str, float, int]] = []
    root_tokens: Counter = Counter()
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
        root_tokens[sop.root.token or "<leaf>"] += 1
        prediction = sop.predict(features[index])
        predictions.append(
            (
                labels[index],
                prediction.verdict or "",
                prediction.confidence_lower_bound,
                prediction.support,
            )
        )

    answered = [item for item in predictions if item[1]]
    correct = sum(1 for truth, verdict, _, _ in answered if truth == verdict)
    gated = [item for item in predictions if item[1] and item[2] >= lower_bound_gate]
    gated_correct = sum(1 for truth, verdict, _, _ in gated if truth == verdict)

    per_label: Dict[str, Dict[str, int]] = {}
    for truth, verdict, _, _ in predictions:
        bucket = per_label.setdefault(truth, {"total": 0, "correct": 0})
        bucket["total"] += 1
        if truth == verdict:
            bucket["correct"] += 1

    return {
        "cases": len(predictions),
        "answered": len(answered),
        "loo_accuracy": round(correct / len(predictions), 4) if predictions else 0.0,
        "loo_accuracy_wilson_lower_bound": wilson_lower_bound(correct, len(predictions)),
        "selective_coverage": round(len(gated) / len(predictions), 4) if predictions else 0.0,
        "selective_precision": round(gated_correct / len(gated), 4) if gated else 0.0,
        "selective_precision_wilson_lower_bound": wilson_lower_bound(gated_correct, len(gated)),
        "per_label_recall": {
            label: round(bucket["correct"] / bucket["total"], 4)
            for label, bucket in sorted(per_label.items())
        },
        "root_split_tokens": dict(root_tokens.most_common()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/rca_v2_l2fixed"))
    parser.add_argument("--feature-profile", default="v2")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-leaf-size", type=int, default=5)
    parser.add_argument("--lower-bound-gate", type=float, default=0.5)
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="只跑指定消融名；不给则全部跑。",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/i1_feature_ablation.json"))
    args = parser.parse_args()

    train_cases = cases_by_manifest_split(args.data_dir, "train")
    labels = [str(case["label"]) for case in train_cases]
    dictionary = dictionary_for(args.feature_profile)
    packs = build_packs(train_cases, source_dataset=str(args.data_dir))
    thresholds = fit_thresholds(train_cases)
    model = fit_feature_model(packs, dictionary=dictionary)
    base_features = [extract_features(pack, thresholds, model, dictionary=dictionary) for pack in packs]

    prior = Counter(labels)
    majority = max(prior, key=lambda label: prior[label])
    print(f"train={len(train_cases)} 多数类={majority} 先验={round(prior[majority]/len(labels),4)}")
    print(f"LOO SOP: max_depth={args.max_depth} min_leaf={args.min_leaf_size} 门限 lb>={args.lower_bound_gate}\n")

    selected = [item for item in ABLATIONS if args.only is None or item[0] in args.only]
    results: Dict[str, Any] = {}
    print(f"{'ablation':>20} {'acc':>7} {'acc_lb':>7} {'cover':>7} {'prec':>7} {'prec_lb':>7}  根分裂 token")
    for name, prefixes in selected:
        masked = [mask(item, prefixes) for item in base_features]
        outcome = leave_one_out(
            masked,
            labels,
            max_depth=args.max_depth,
            min_leaf_size=args.min_leaf_size,
            lower_bound_gate=args.lower_bound_gate,
        )
        outcome["dropped_prefixes"] = list(prefixes)
        results[name] = outcome
        roots = "; ".join(f"{token}×{count}" for token, count in list(outcome["root_split_tokens"].items())[:2])
        print(
            f"{name:>20} {outcome['loo_accuracy']:>7.4f}"
            f" {outcome['loo_accuracy_wilson_lower_bound']:>7.4f}"
            f" {outcome['selective_coverage']:>7.4f} {outcome['selective_precision']:>7.4f}"
            f" {outcome['selective_precision_wilson_lower_bound']:>7.4f}  {roots}"
        )

    report = {
        "schema_version": "feature-ablation-v1",
        "data_dir": str(args.data_dir),
        "feature_dictionary_version": dictionary.version,
        "train_size": len(train_cases),
        "majority_baseline": round(prior[majority] / len(labels), 4),
        "sop": {"max_depth": args.max_depth, "min_leaf_size": args.min_leaf_size},
        "lower_bound_gate": args.lower_bound_gate,
        "protocol": "留一法：每条 case 都用其余 n-1 条重学 SOP 后预测，标签只来自 train split。",
        "ablations": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
