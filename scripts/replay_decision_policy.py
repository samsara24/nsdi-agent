#!/usr/bin/env python3
"""离线复算 M9 决策策略，不重跑 LLM、不占用 GPU。

正式实验已经把每个 case 的 `branch_outcome`、`sop_prediction`、`match` 和真值写进
`outcomes.json`。M9 只是在这些量上做一次阈值判定，因此换阈值不需要重新推理。
本脚本据此回答三个问题：

1. 阈值放到什么位置才会有 case 通过 M9，通过后的准确率与人工介入率是多少。
2. 自动结论的候选应该取自哪一路：分支结论、learned SOP、历史近邻还是多数类。
3. 各分支在「强制回答」口径下的天花板，用于判断瓶颈在阈值还是在候选质量。

所有数字都是对既有产物的描述性复算，不是新的方法结论。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

ROOT_CAUSES = ("L1", "L2", "fiber")


def wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    phat = successes / total
    denominator = 1.0 + z * z / total
    centre = phat + z * z / (2 * total)
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4 * total)) / total)
    return round(max(0.0, (centre - margin) / denominator), 6)


def load_outcomes(run_dir: Path, policy: str) -> List[Mapping[str, Any]]:
    payload = json.loads((run_dir / "outcomes.json").read_text(encoding="utf-8"))
    if policy not in payload:
        raise SystemExit(f"policy {policy!r} not in outcomes.json; available: {sorted(payload)}")
    return payload[policy]


def candidate_from(record: Mapping[str, Any], source: str, train_prior: str) -> Optional[str]:
    """按不同候选来源给出该 case 的建议结论。"""
    branch = record["branch_outcome"].get("verdict")
    sop = (record.get("sop_prediction") or {}).get("verdict")
    candidates = record["match"].get("candidates") or []
    neighbour = candidates[0]["label"] if candidates else None

    if source == "branch":
        return branch
    if source == "sop":
        return sop
    if source == "neighbour":
        return neighbour
    if source == "majority":
        return train_prior
    if source == "branch_then_sop":
        return branch if branch is not None else sop
    if source == "branch_then_neighbour_then_sop":
        if branch is not None:
            return branch
        if neighbour is not None:
            return neighbour
        return sop
    raise ValueError(f"unknown candidate source: {source}")


def score(predictions: Sequence[Optional[str]], truths: Sequence[str]) -> Dict[str, Any]:
    answered = [(p, t) for p, t in zip(predictions, truths) if p is not None]
    correct = sum(1 for p, t in answered if p == t)
    per_class: Dict[str, Dict[str, Any]] = {}
    for label in ROOT_CAUSES:
        support = sum(1 for t in truths if t == label)
        predicted = sum(1 for p, _ in answered if p == label)
        tp = sum(1 for p, t in answered if p == label and t == label)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "support": support,
            "predicted": predicted,
            "true_positive": tp,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
    return {
        "total": len(truths),
        "answered": len(answered),
        "coverage": round(len(answered) / len(truths), 6) if truths else 0.0,
        "correct": correct,
        "precision_when_answered": round(correct / len(answered), 6) if answered else None,
        "accuracy_over_all": round(correct / len(truths), 6) if truths else 0.0,
        "macro_f1": round(sum(per_class[c]["f1"] for c in ROOT_CAUSES) / len(ROOT_CAUSES), 6),
        "per_class": per_class,
    }


def replay_gate(
    records: Sequence[Mapping[str, Any]],
    *,
    lower_bound: float,
    minimum_support: int,
    source: str,
    train_prior: str,
    fallback: Optional[str],
) -> Dict[str, Any]:
    """按给定阈值复算 M9 出口。

    `fallback` 不为空时，被门禁拒绝的 case 会用该来源的候选作为「降级建议」，
    用于回答「人工不介入也给个默认答案时整体准确率是多少」。
    """
    truths = [record["actual"] for record in records]
    auto: List[Optional[str]] = []
    actions = Counter()
    for record in records:
        outcome = record["branch_outcome"]
        verdict = candidate_from(record, source, train_prior)
        support = int(outcome.get("calibration_support", 0))
        bound = float(outcome.get("confidence_lower_bound", 0.0))
        if source in {"sop", "majority"}:
            sop = record.get("sop_prediction") or {}
            if source == "sop":
                support = int(sop.get("support", 0))
                bound = float(sop.get("confidence_lower_bound", 0.0))
        passes = verdict is not None and support >= minimum_support and bound >= lower_bound
        if passes:
            actions["final"] += 1
            auto.append(verdict)
        else:
            auto.append(None)
            actions["request_evidence" if outcome.get("missing_evidence") else "human_review"] += 1

    gated = score(auto, truths)
    result = {
        "policy": {
            "final_lower_bound": lower_bound,
            "minimum_support": minimum_support,
            "candidate_source": source,
        },
        "actions": dict(actions),
        "human_intervention_rate": round(actions["human_review"] / len(records), 6),
        "request_evidence_rate": round(actions["request_evidence"] / len(records), 6),
        "gated": gated,
    }
    if fallback:
        merged = [
            a if a is not None else candidate_from(record, fallback, train_prior)
            for a, record in zip(auto, records)
        ]
        result["with_fallback"] = {"source": fallback, **score(merged, truths)}
    return result


def branch_ceiling(records: Sequence[Mapping[str, Any]], train_prior: str) -> Dict[str, Any]:
    """每个分支在「强制回答」口径下各候选来源的准确率。"""
    sources = ("branch", "sop", "neighbour", "majority", "branch_then_neighbour_then_sop")
    by_branch: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_branch[record["branch_outcome"]["branch"]].append(record)

    report: Dict[str, Any] = {}
    for branch in sorted(by_branch) + ["ALL"]:
        subset = records if branch == "ALL" else by_branch[branch]
        truths = [record["actual"] for record in subset]
        entry: Dict[str, Any] = {"n": len(subset), "label_distribution": dict(Counter(truths))}
        for source in sources:
            preds = [candidate_from(record, source, train_prior) for record in subset]
            stats = score(preds, truths)
            entry[source] = {
                "answered": stats["answered"],
                "correct": stats["correct"],
                "accuracy_over_all": stats["accuracy_over_all"],
                "precision_when_answered": stats["precision_when_answered"],
            }
        oracle = sum(
            1
            for record in subset
            if record["actual"]
            in {
                candidate_from(record, s, train_prior)
                for s in ("branch", "sop", "neighbour")
            }
        )
        entry["oracle_over_sources"] = {
            "correct": oracle,
            "accuracy_over_all": round(oracle / len(subset), 6) if subset else 0.0,
        }
        report[branch] = entry
    return report


def calibration_groups(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """测试集上每个标定分组的实际表现，用来核对训练标定是否可外推。"""
    tally: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0])
    claimed: Dict[str, float] = {}
    for record in records:
        outcome = record["branch_outcome"]
        group = outcome.get("calibration_group", "")
        row = tally[group]
        row[2] += 1
        if outcome.get("verdict") is not None:
            row[1] += 1
            row[0] += int(outcome["verdict"] == record["actual"])
        claimed[group] = float(outcome.get("confidence_lower_bound", 0.0))
    return {
        group: {
            "cases": row[2],
            "answered": row[1],
            "correct": row[0],
            "observed_precision": round(row[0] / row[1], 6) if row[1] else None,
            "observed_wilson_lower_bound": wilson_lower_bound(row[0], row[1]),
            "train_claimed_lower_bound": round(claimed[group], 6),
        }
        for group, row in sorted(tally.items())
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("artifacts/offline_sop_llm_l2fixed_deepseek32b_seed42_promptv6"),
    )
    parser.add_argument("--policy", default="coverage-v2")
    parser.add_argument("--train-prior", default="L2", help="训练集多数类，用于 majority 基线")
    parser.add_argument("--fallback", default="sop", help="被门禁拒绝时的降级候选来源；空串表示不降级")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    records = load_outcomes(args.run_dir, args.policy)
    truths = [record["actual"] for record in records]

    sweep = []
    for source in ("branch", "sop", "branch_then_sop", "branch_then_neighbour_then_sop"):
        for bound in (0.0, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6):
            for support in (1, 5, 10, 20):
                sweep.append(
                    replay_gate(
                        records,
                        lower_bound=bound,
                        minimum_support=support,
                        source=source,
                        train_prior=args.train_prior,
                        fallback=args.fallback or None,
                    )
                )

    report = {
        "schema_version": "decision-replay-v1",
        "run_dir": str(args.run_dir),
        "policy": args.policy,
        "case_count": len(records),
        "label_distribution": dict(Counter(truths)),
        "majority_baseline": score([args.train_prior] * len(records), truths),
        "branch_ceiling": branch_ceiling(records, args.train_prior),
        "test_side_calibration": calibration_groups(records),
        "gate_sweep": sweep,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"case_count={len(records)}  labels={report['label_distribution']}")
    print(f"majority({args.train_prior}) accuracy = {report['majority_baseline']['accuracy_over_all']:.4f}")
    print()
    print("== 强制回答口径下各分支的候选来源准确率 ==")
    header = f"{'branch':>6} {'n':>4} {'branch':>8} {'sop':>8} {'neighbour':>10} {'majority':>9} {'cascade':>8} {'oracle':>7}"
    print(header)
    for branch, entry in report["branch_ceiling"].items():
        print(
            f"{branch:>6} {entry['n']:>4}"
            f" {entry['branch']['correct']:>8}"
            f" {entry['sop']['correct']:>8}"
            f" {entry['neighbour']['correct']:>10}"
            f" {entry['majority']['correct']:>9}"
            f" {entry['branch_then_neighbour_then_sop']['correct']:>8}"
            f" {entry['oracle_over_sources']['correct']:>7}"
        )
    print()
    print("== M9 阈值扫描（只显示有 final 的行）==")
    print(
        f"{'source':>32} {'lb':>5} {'sup':>4} {'final':>6} {'corr':>5} {'prec':>7} {'human%':>7} {'fb_acc':>7}"
    )
    for row in sweep:
        if row["actions"].get("final", 0) == 0:
            continue
        fb = row.get("with_fallback", {})
        print(
            f"{row['policy']['candidate_source']:>32}"
            f" {row['policy']['final_lower_bound']:>5.2f}"
            f" {row['policy']['minimum_support']:>4}"
            f" {row['actions']['final']:>6}"
            f" {row['gated']['correct']:>5}"
            f" {(row['gated']['precision_when_answered'] or 0):>7.4f}"
            f" {row['human_intervention_rate']:>7.4f}"
            f" {fb.get('accuracy_over_all', 0):>7.4f}"
        )
    if args.output:
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
