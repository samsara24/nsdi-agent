#!/usr/bin/env python3
"""在 rca_v2_l2fixed 的 161/107 划分上评测专家决策树。

这个脚本回答一个迭代 3 必须先问清楚的问题：**把现网专家经验原样搬过来，
在这份数据上到底值多少？** 在此之前所有路线（浅决策树、门限校准、相似度投票）
相对多数类的增益都不超过 +2.3pp，而三条独立路线测出的可辨识上限是 70~75%。

报告口径遵循迭代 2 定下的规矩，不能只报准确率：

- 每一档都必须带**同一批 case 上的多数类基线**，否则「精度高」可能只是
  「这批 case 恰好多数类富集」。
- 必须报**平衡召回**。一律报 L2 的准确率是 62.6%，但平衡召回只有 1/3，
  上一轮就是靠这个指标识破了退化解。
- 必须按**裁决来源分桶**。专家规则里 `no_anomaly` 与 `port_status_gate`
  是兜底出口而不是判别，把它们混进总准确率会虚高。

专家规则不在本数据上拟合任何参数，所以 train 与 test 的差异只反映分布差异。
两个划分都报，是为了让这一点可被检验，而不是为了挑好看的那个。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca_framework.data import cases_by_manifest_split  # noqa: E402
from rca_framework.evidence_pack import EvidencePack, build_packs, labels_of  # noqa: E402
from rca_framework.expert import (  # noqa: E402
    DOC_VARIANT,
    EXPERT_RULES_VERSION,
    SINGLE_METRIC_DIRECTION,
    ExpertCalibration,
    ExpertVariant,
    diagnose_many,
)
from rca_framework.types import ROOT_CAUSES  # noqa: E402


def _flip(direction: str) -> str:
    return "same" if direction == "opposite" else "opposite"


#: 消融组。每一项只动一个维度，用来把 +14pp 拆到具体知识上。
ABLATIONS: Dict[str, ExpertVariant] = {
    "doc": DOC_VARIANT,
    # 方向表整体反转。若增益来自「归因方向」这份知识，这一组应当显著低于多数类。
    "reverse_direction": ExpertVariant(
        name="reverse_direction",
        single_metric_direction={k: _flip(v) for k, v in SINGLE_METRIC_DIRECTION.items()},
        multi_metric_direction="same",
        txpower_lane_down_direction="opposite",
    ),
    # 所有异常都归本端 / 都归对端。检验方向表是否只是「一律指向某一端」的复杂写法。
    "always_same_side": ExpertVariant(
        name="always_same_side",
        single_metric_direction={k: "same" for k in SINGLE_METRIC_DIRECTION},
        multi_metric_direction="same",
        txpower_lane_down_direction="same",
    ),
    "always_opposite_side": ExpertVariant(
        name="always_opposite_side",
        single_metric_direction={k: "opposite" for k in SINGLE_METRIC_DIRECTION},
        multi_metric_direction="opposite",
        txpower_lane_down_direction="opposite",
    ),
    # 不解析告警端，一律把 L2 当本端。检验增益是否来自 alarm_ip_interface。
    "no_alarm_resolution": ExpertVariant(name="no_alarm_resolution", resolve_alarm_side=False),
    # 关掉两个兜底出口。剩下的就是纯证据驱动的判别力。
    "no_fallback": ExpertVariant(name="no_fallback", use_fallbacks=False),
}


def evaluate(
    packs: Sequence[EvidencePack],
    labels: Sequence[str],
    *,
    split: str,
    variant: ExpertVariant = DOC_VARIANT,
) -> Dict[str, Any]:
    diagnoses = diagnose_many(packs, variant=variant)
    matrix: Dict[str, Counter] = {label: Counter() for label in ROOT_CAUSES}
    by_group: Dict[str, List[int]] = {}
    rows: List[Dict[str, Any]] = []
    unresolved = 0

    for pack, diagnosis, truth in zip(packs, diagnoses, labels):
        verdict = diagnosis.verdict
        matrix[truth][verdict or "abstain"] += 1
        entry = by_group.setdefault(diagnosis.group, [0, 0])
        entry[0] += int(verdict == truth)
        entry[1] += 1
        unresolved += int(not diagnosis.alarm_side_resolved)
        rows.append(
            {
                "case_id": pack.case_id,
                "label": truth,
                "verdict": verdict,
                "group": diagnosis.group,
                "priority": diagnosis.priority,
                "alarm_side_resolved": diagnosis.alarm_side_resolved,
            }
        )

    total = len(labels)
    correct = sum(matrix[label][label] for label in ROOT_CAUSES)
    counts = Counter(labels)
    majority_label, majority_n = counts.most_common(1)[0]

    per_class = {}
    recalls = []
    for label in ROOT_CAUSES:
        support = counts.get(label, 0)
        predicted = sum(matrix[other][label] for other in ROOT_CAUSES)
        hit = matrix[label][label]
        recall = hit / support if support else 0.0
        precision = hit / predicted if predicted else 0.0
        per_class[label] = {
            "support": support,
            "predicted": predicted,
            "recall": round(recall, 6),
            "precision": round(precision, 6),
        }
        if support:
            recalls.append(recall)

    # 判别桶：剔除两个兜底出口后剩下的、真正由证据驱动的裁决。
    fallback_groups = {"expert:no_anomaly", "expert:port_status_gate"}
    discriminative = [row for row in rows if row["group"] not in fallback_groups]
    disc_truth = Counter(row["label"] for row in discriminative)
    disc_majority = max(disc_truth.values()) if disc_truth else 0
    disc_correct = sum(1 for row in discriminative if row["verdict"] == row["label"])

    answered = sum(1 for row in rows if row["verdict"] is not None)
    answered_correct = sum(1 for row in rows if row["verdict"] == row["label"])
    answered_majority = 0
    if answered:
        answered_truth = Counter(row["label"] for row in rows if row["verdict"] is not None)
        answered_majority = max(answered_truth.values())

    return {
        "split": split,
        "variant": variant.name,
        "version": EXPERT_RULES_VERSION,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 6) if total else 0.0,
        "answered": answered,
        "coverage": round(answered / total, 6) if total else 0.0,
        "precision_when_answered": round(answered_correct / answered, 6) if answered else 0.0,
        "majority_on_answered": round(answered_majority / answered, 6) if answered else 0.0,
        "lift_on_answered": round((answered_correct - answered_majority) / answered, 6) if answered else 0.0,
        "majority_label": majority_label,
        "majority_baseline": round(majority_n / total, 6) if total else 0.0,
        "lift_over_majority": round((correct - majority_n) / total, 6) if total else 0.0,
        "balanced_recall": round(sum(recalls) / len(recalls), 6) if recalls else 0.0,
        "alarm_side_unresolved": unresolved,
        "confusion_matrix": {label: dict(matrix[label]) for label in ROOT_CAUSES},
        "per_class": per_class,
        "discriminative_subset": {
            "n": len(discriminative),
            "correct": disc_correct,
            "accuracy": round(disc_correct / len(discriminative), 6) if discriminative else 0.0,
            "majority_baseline": round(disc_majority / len(discriminative), 6) if discriminative else 0.0,
            "lift": round((disc_correct - disc_majority) / len(discriminative), 6) if discriminative else 0.0,
        },
        "by_group": {
            group: {
                "correct": value[0],
                "total": value[1],
                "accuracy": round(value[0] / value[1], 6) if value[1] else 0.0,
            }
            for group, value in sorted(by_group.items(), key=lambda item: -item[1][1])
        },
        "rows": rows,
    }


def print_report(result: Dict[str, Any]) -> None:
    print(f"\n{'=' * 72}")
    print(
        f"{result['split']}  n={result['total']}  "
        f"准确率 {result['correct']}/{result['total']} = {result['accuracy']:.2%}  "
        f"多数类 {result['majority_baseline']:.2%}  lift {result['lift_over_majority']:+.2%}"
    )
    print(f"平衡召回 {result['balanced_recall']:.4f}   告警端未解析 {result['alarm_side_unresolved']} 条")
    print("=" * 72)
    header = f"{'真值\\预测':<10}" + "".join(f"{label:>8}" for label in ROOT_CAUSES) + f"{'recall':>10}"
    print(header)
    for label in ROOT_CAUSES:
        row = result["confusion_matrix"][label]
        cells = "".join(f"{row.get(other, 0):>8}" for other in ROOT_CAUSES)
        print(f"{label:<10}{cells}{result['per_class'][label]['recall']:>10.2%}")

    disc = result["discriminative_subset"]
    print(
        f"\n剔除兜底出口后的判别子集：{disc['correct']}/{disc['n']} = {disc['accuracy']:.2%}"
        f"，同子集多数类 {disc['majority_baseline']:.2%}，lift {disc['lift']:+.2%}"
    )
    print("\n按裁决分组：")
    for group, stats in result["by_group"].items():
        print(f"  {group:<34}{stats['total']:>4} 条  命中 {stats['correct']:>3}  {stats['accuracy']:>7.2%}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="datasets/rca_v2_l2fixed")
    parser.add_argument("--output", default="artifacts/i3_expert_rules_l2fixed.json")
    parser.add_argument("--skip-ablations", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    summary: Dict[str, Any] = {"version": EXPERT_RULES_VERSION, "data_dir": str(data_dir)}
    train_packs: List[EvidencePack] = []
    train_labels: List[str] = []

    for split in ("train", "test"):
        cases = cases_by_manifest_split(data_dir, split)
        packs = build_packs(cases, source_dataset=str(data_dir))
        labels = labels_of(cases)
        result = evaluate(packs, labels, split=split)
        print_report(result)
        summary[split] = {key: value for key, value in result.items() if key != "rows"}
        summary.setdefault("rows", {})[split] = result["rows"]
        if split == "train":
            train_packs, train_labels = list(packs), list(labels)

    if not args.skip_ablations:
        print(f"\n{'=' * 72}\n消融：每组只改一个维度，全部在 test 上评（专家规则不拟合参数）\n{'=' * 72}")
        print(
            f"{'变体':<24}{'覆盖':>8}{'答题精度':>10}{'同子集多数类':>14}{'lift':>9}{'平衡召回':>10}"
        )
        cases = cases_by_manifest_split(data_dir, "test")
        packs = build_packs(cases, source_dataset=str(data_dir))
        labels = labels_of(cases)
        ablation_summary = {}
        for name, variant in ABLATIONS.items():
            result = evaluate(packs, labels, split="test", variant=variant)
            ablation_summary[name] = {
                key: value for key, value in result.items() if key not in ("rows", "by_group")
            }
            print(
                f"{name:<24}{result['coverage']:>8.2%}{result['precision_when_answered']:>10.2%}"
                f"{result['majority_on_answered']:>14.2%}{result['lift_on_answered']:>+9.2%}"
                f"{result['balanced_recall']:>10.4f}"
            )
        summary["ablations_test"] = ablation_summary

    calibration = ExpertCalibration.fit(
        diagnose_many(train_packs), train_labels, source="train-in-sample"
    )
    summary["calibration_train_in_sample"] = calibration.to_dict()
    print("\n训练集分组标定（供 M9 使用，Wilson 95% 下界）：")
    for group, stats in calibration.to_dict()["groups"].items():
        print(
            f"  {group:<34}{stats['total']:>4} 条  精度 {stats['accuracy']:>7.2%}  "
            f"下界 {stats['wilson_lower_bound']:>7.4f}"
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果写入 {out_path}")


if __name__ == "__main__":
    main()
