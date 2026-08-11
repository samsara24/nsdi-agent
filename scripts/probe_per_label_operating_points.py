"""迭代 2 离线探针：每个类别的候选到底能不能单独达标。

迭代 1 的结论是「统一门限把 L1 全挡了」，但这句话有两种完全不同的成因，
处置方式相反：

1. **门限表达能力不足**：L1 候选里存在一段纯度达标的子集，只是它的置信度下界
   低于 L2 那段不可用候选，单一标量无法同时收 L1 的好子集、拒 L2 的坏子集。
   这种情况按类别校准就能救回来。
2. **候选本身不可分**：L1 候选在任何门限下纯度都达不到目标，低置信度不是
   校准问题而是如实反映了候选质量。这种情况调门限只能在「不答」和
   「答错」之间选，真正要改的是上游候选生成。

本脚本在训练留一法上把每个类别的风险-覆盖率曲线单独打出来，用来判断是哪一种。
不接 LLM、不用 GPU，也绝不读测试集标签。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca_framework.anomaly import fit_thresholds
from rca_framework.branches import fit_calibration, handle_many
from rca_framework.data import cases_by_manifest_split
from rca_framework.decision import (
    DecisionCandidate,
    DecisionPolicy,
    build_candidates,
    simulate_gate,
)
from rca_framework.evidence_graph import BOARD_POLICY, COVERAGE_POLICY, EvidenceGraph, match_many
from rca_framework.evidence_pack import build_packs, labels_of
from rca_framework.features.dictionary import dictionary_for
from rca_framework.features.extractor import extract_features, fit_feature_model
from rca_framework.knowledge import _loo_sop_predictions, _out_of_fold_sop_predictions
from rca_framework.sop import learn_sop


def train_rows(
    data_dir: Path,
    *,
    feature_profile: str,
    policy_name: str,
    minimum_support: int,
    candidate_order: Tuple[str, ...],
    sop_confidence: str = "out-of-fold",
    folds: int = 5,
) -> Tuple[List[Tuple[Sequence[DecisionCandidate], str]], Dict[str, int]]:
    train_cases = cases_by_manifest_split(data_dir, "train")
    labels = labels_of(train_cases)
    dictionary = dictionary_for(feature_profile)
    thresholds = fit_thresholds(train_cases)
    packs = build_packs(train_cases, source_dataset=str(data_dir))
    model = fit_feature_model(packs, dictionary=dictionary)
    features = [extract_features(pack, thresholds, model, dictionary=dictionary) for pack in packs]
    sop = learn_sop(features, labels, source=f"{data_dir.name}:manifest-train")
    graph = EvidenceGraph.build(
        features, labels, feature_model=model, dictionary=dictionary, source_dataset=str(data_dir)
    )
    results = match_many(graph, features, top_k=0, leave_one_out=True)
    policy = {BOARD_POLICY.name: BOARD_POLICY, COVERAGE_POLICY.name: COVERAGE_POLICY}[policy_name]
    calibration = fit_calibration(
        results, packs, labels, policy=policy, source="manifest-train-loo"
    )
    paired = handle_many(
        results, packs, calibration, policy=policy, reasoner=None,
        features=features, sop_model=sop,
    )
    outcomes = [item[1] for item in paired]
    if sop_confidence == "loo":
        loo_sop = _loo_sop_predictions(features, labels, sop=sop)
    else:
        loo_sop = _out_of_fold_sop_predictions(features, labels, sop=sop, folds=folds)
    probe = DecisionPolicy(
        final_lower_bound=0.0, minimum_support=minimum_support, candidate_order=candidate_order
    )
    rows = [
        (build_candidates(outcome, sop_prediction=sop_pred, policy=probe), truth)
        for outcome, sop_pred, truth in zip(outcomes, loo_sop, labels)
    ]
    prior: Dict[str, int] = {}
    for label in labels:
        prior[label] = prior.get(label, 0) + 1
    return rows, prior


def per_label_curve(
    rows: Sequence[Tuple[Sequence[DecisionCandidate], str]],
    label: str,
    *,
    minimum_support: int,
    candidate_order: Tuple[str, ...],
) -> List[Dict[str, float]]:
    """只看指向 `label` 的候选：随着自身门限下降，纯度如何变化。

    其它类别的门限固定为 1.0（Wilson 下界在有限样本上永远达不到 1.0，等价于全拒），
    这样曲线只反映该类候选自身的质量，不掺入别的类别的正确率。
    """
    others = {
        str(candidate.verdict): 1.0
        for candidates, _ in rows
        for candidate in candidates
        if candidate.verdict is not None and str(candidate.verdict) != label
    }
    thresholds = sorted(
        {
            round(candidate.confidence_lower_bound, 6)
            for candidates, _ in rows
            for candidate in candidates
            if str(candidate.verdict) == label and candidate.support >= minimum_support
        }
        | {0.0}
    )
    curve = []
    for threshold in thresholds:
        bounds = dict(others)
        bounds[label] = threshold
        policy = DecisionPolicy(
            final_lower_bound=1.0,
            minimum_support=minimum_support,
            candidate_order=candidate_order,
            per_label_lower_bound=bounds,
        )
        stats = simulate_gate(rows, policy)
        row = stats["by_predicted_label"].get(label)
        if not row:
            continue
        curve.append(
            {
                "lower_bound": threshold,
                "answered": row["answered"],
                "correct": row["correct"],
                "precision": row["precision"],
                "selective_risk": row["selective_risk"],
            }
        )
    return curve


def confidence_auc(pairs: Sequence[Tuple[float, int]]) -> float | None:
    """置信度把「判对」排在「判错」前面的概率（并列算半分）。

    0.5 表示置信度与正确性无关，> 0.5 表示有区分力，
    **< 0.5 表示反序**——门限用它就会反向筛选。
    """
    positives = [value for value, hit in pairs if hit]
    negatives = [value for value, hit in pairs if not hit]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def inversion_report(
    rows: Sequence[Tuple[Sequence[DecisionCandidate], str]],
    *,
    source: str,
    minimum_support: int,
) -> Dict[str, object]:
    """按标定分组检查「置信度 vs 正确性」的排序方向。

    必须**在分组内**看：跨组比较会被不同叶子的真实纯度差异掩盖，
    而门限恰恰是跨组的一条线，所以组内反序会直接变成门限的反向筛选。
    """
    grouped: Dict[str, List[Tuple[float, int]]] = {}
    pooled: List[Tuple[float, int]] = []
    for candidates, truth in rows:
        for candidate in candidates:
            if candidate.source != source or candidate.support < minimum_support:
                continue
            hit = int(candidate.verdict == truth)
            grouped.setdefault(candidate.group, []).append(
                (candidate.confidence_lower_bound, hit)
            )
            pooled.append((candidate.confidence_lower_bound, hit))
    return {
        "pooled_auc": confidence_auc(pooled),
        "by_group": {
            group: {
                "n": len(pairs),
                "auc": confidence_auc(pairs),
                "distinct_lower_bounds": len({round(value, 6) for value, _ in pairs}),
            }
            for group, pairs in sorted(grouped.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/rca_v2_l2fixed"))
    parser.add_argument("--feature-profile", default="v2")
    parser.add_argument("--policy", default="coverage-v2")
    parser.add_argument("--decision-min-support", type=int, default=10)
    parser.add_argument("--decision-candidate-order", nargs="+", default=("branch", "sop"))
    parser.add_argument(
        "--sop-confidence",
        choices=("loo", "out-of-fold"),
        default="out-of-fold",
        help="SOP 候选置信度的来源；loo 仅用于复现迭代 1 的反序现象",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows, prior = train_rows(
        args.data_dir.resolve(),
        feature_profile=args.feature_profile,
        policy_name=args.policy,
        minimum_support=args.decision_min_support,
        candidate_order=tuple(args.decision_candidate_order),
        sop_confidence=args.sop_confidence,
        folds=args.folds,
    )
    print(f"sop confidence: {args.sop_confidence}" + (f"（{args.folds} 折）" if args.sop_confidence != "loo" else ""))
    total = sum(prior.values())
    print(f"train cases   : {total}")
    print("label prior   : " + "，".join(
        f"{label} {count}（{count / total:.2%}）" for label, count in sorted(prior.items())
    ))

    labels = sorted(
        {
            str(candidate.verdict)
            for candidates, _ in rows
            for candidate in candidates
            if candidate.verdict is not None
        }
    )
    inversions = {
        source: inversion_report(rows, source=source, minimum_support=args.decision_min_support)
        for source in sorted({candidate.source for candidates, _ in rows for candidate in candidates})
    }
    print("\n===== 置信度排序方向（组内 AUC，< 0.5 即反序）=====")
    for source, item in inversions.items():
        pooled = item["pooled_auc"]
        print(f"  {source:8s} 合并 AUC {'—' if pooled is None else f'{pooled:.4f}'}")
        for group, stat in item["by_group"].items():
            auc = stat["auc"]
            if auc is None:
                continue
            flag = " <== 反序" if auc < 0.5 else ""
            print(f"    {group:34s} n={stat['n']:3d} 取值数={stat['distinct_lower_bounds']:2d} AUC={auc:.4f}{flag}")

    report: Dict[str, object] = {"prior": prior, "curves": {}, "inversion": inversions}
    for label in labels:
        curve = per_label_curve(
            rows,
            label,
            minimum_support=args.decision_min_support,
            candidate_order=tuple(args.decision_candidate_order),
        )
        report["curves"][label] = curve
        share = prior.get(label, 0) / total if total else 0.0
        print(f"\n===== 预测为 {label} 的候选（该类先验 {share:.2%}）=====")
        print("  下界      作答   判对    纯度      风险")
        for item in curve:
            precision = item["precision"]
            risk = item["selective_risk"]
            print(
                f"  {item['lower_bound']:.4f}  {item['answered']:4d}  {item['correct']:4d}  "
                f"{precision:7.2%}  {risk:7.2%}"
            )
        best = max(
            (item for item in curve if item["answered"] >= args.decision_min_support),
            key=lambda item: item["precision"],
            default=None,
        )
        if best is not None:
            print(
                f"  该类在作答数 >= {args.decision_min_support} 时的最高纯度："
                f"{best['precision']:.2%}（下界 {best['lower_bound']:.4f}，作答 {best['answered']}）"
            )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
