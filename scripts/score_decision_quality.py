#!/usr/bin/env python3
"""对已有实验产物做同口径复盘，补上三个能识破退化解的指标。

现有的 `answer_rate` / `precision_when_answered` 有一个共同盲区：它们无法区分
「系统真的诊断出了根因」和「系统在多数类上蒙对了」。在 rca_v2_l2fixed 上
一律报 L2 就有 62.6% 的测试准确率与 0% 人工干预，能同时打赢覆盖率和准确率两项，
却对 L1 与 fiber 毫无用处。所以本脚本额外算三项：

1. `lift_over_majority_on_kept`：被保留的那批 case 上，系统精度减去同一批
   case 上一律报多数类的精度。必须同子集比，否则测的只是「门限挑走了容易的 case」。
2. `balanced_recall`：三类召回的算术平均。多数类蒙对拉不动这个数。
3. `abstention_effectiveness`：被弃答（转人工 / 补采）的 case 里，
   如果强行用 SOP 兜底作答会答错的比例。它回答的是「人工是否被用在了对的地方」——
   降低人工比例只有在弃答仍然精准时才是改进，否则只是把错误推给自动结论。

用法::

    python scripts/score_decision_quality.py artifacts/<run>/ [artifacts/<other-run>/ ...]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca_framework.branches.base import wilson_lower_bound  # noqa: E402
from rca_framework.types import ROOT_CAUSES  # noqa: E402

#: 视为「给出了自动结论」的 M9 动作，与 `rca_framework.decision` 的 `ACTIONS` 保持一致。
#: 其余动作（request_evidence / human_review）都算弃答。
ANSWER_ACTIONS = ("final",)


def load_outcomes(run_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    payload = json.loads((run_dir / "outcomes.json").read_text(encoding="utf-8"))
    if isinstance(payload, dict) and not any(isinstance(value, list) for value in payload.values()):
        return {"default": [payload]}
    return {str(key): list(value) for key, value in payload.items()}


def verdict_of(case: Dict[str, Any]) -> Optional[str]:
    decision = case.get("final_decision") or {}
    if decision.get("action") in ANSWER_ACTIONS:
        return decision.get("verdict")
    return None


def sop_verdict_of(case: Dict[str, Any]) -> Optional[str]:
    prediction = case.get("sop_prediction") or {}
    return prediction.get("verdict")


def score(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    truths = [str(case.get("actual")) for case in cases]
    counts = Counter(truths)
    majority = max(counts, key=lambda label: counts[label]) if counts else ""

    answered = [case for case in cases if verdict_of(case)]
    abstained = [case for case in cases if not verdict_of(case)]
    answered_correct = sum(1 for case in answered if verdict_of(case) == str(case.get("actual")))
    majority_on_kept = sum(1 for case in answered if str(case.get("actual")) == majority)

    # 弃答有效性：这些 case 若用 SOP 兜底，会错多少。
    forced = [case for case in abstained if sop_verdict_of(case)]
    forced_wrong = sum(1 for case in forced if sop_verdict_of(case) != str(case.get("actual")))

    per_label: Dict[str, Dict[str, Any]] = {}
    for label in ROOT_CAUSES:
        total = counts.get(label, 0)
        hit = sum(
            1
            for case in cases
            if str(case.get("actual")) == label and verdict_of(case) == label
        )
        per_label[label] = {
            "support": total,
            "recall": round(hit / total, 4) if total else None,
        }
    recalls = [value["recall"] for value in per_label.values() if value["recall"] is not None]

    sop_all = [case for case in cases if sop_verdict_of(case)]
    sop_correct = sum(1 for case in sop_all if sop_verdict_of(case) == str(case.get("actual")))

    action_counts = Counter(
        (case.get("final_decision") or {}).get("action", "missing") for case in cases
    )

    return {
        "cases": len(cases),
        "majority_label": majority,
        "majority_rate": round(counts.get(majority, 0) / len(cases), 4) if cases else 0.0,
        "answered": len(answered),
        "coverage": round(len(answered) / len(cases), 4) if cases else 0.0,
        "manual_or_evidence_rate": round(len(abstained) / len(cases), 4) if cases else 0.0,
        "precision_when_answered": round(answered_correct / len(answered), 4) if answered else None,
        "precision_wilson_lower_bound": (
            wilson_lower_bound(answered_correct, len(answered)) if answered else None
        ),
        "majority_on_kept": round(majority_on_kept / len(answered), 4) if answered else None,
        "lift_over_majority_on_kept": (
            round((answered_correct - majority_on_kept) / len(answered), 4) if answered else None
        ),
        "balanced_recall": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
        "per_label": per_label,
        "abstention_effectiveness": {
            "abstained": len(abstained),
            "with_sop_fallback": len(forced),
            "sop_would_be_wrong": forced_wrong,
            "effectiveness": round(forced_wrong / len(forced), 4) if forced else None,
            "note": (
                "弃答集里 SOP 兜底会答错的比例。高于保留集的错误率才说明人工被用在了对的地方。"
            ),
        },
        "sop_reference": {
            "predicted": len(sop_all),
            "correct": sop_correct,
            "accuracy_over_all_cases": round(sop_correct / len(cases), 4) if cases else 0.0,
        },
        "majority_reference": {
            "accuracy_over_all_cases": round(counts.get(majority, 0) / len(cases), 4) if cases else 0.0,
            "manual_rate": 0.0,
            "balanced_recall": round((1.0 / len(ROOT_CAUSES)), 4),
            "note": "一律报多数类、从不弃答。它的 balanced_recall 上限就是 1/类别数。",
        },
        "action_breakdown": dict(action_counts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report: Dict[str, Any] = {"schema_version": "decision-quality-v1", "runs": {}}
    for run_dir in args.run_dirs:
        outcomes = load_outcomes(run_dir)
        report["runs"][str(run_dir)] = {}
        for policy, cases in outcomes.items():
            stats = score(cases)
            report["runs"][str(run_dir)][policy] = stats
            abstention = stats["abstention_effectiveness"]
            print(f"── {run_dir.name} / {policy}  ({stats['cases']} case)")
            print(
                f"   覆盖 {stats['coverage']:.4f} | 人工或补采 {stats['manual_or_evidence_rate']:.4f}"
                f" | 给结论时精度 {stats['precision_when_answered']}"
            )
            print(
                f"   同子集多数类 {stats['majority_on_kept']}"
                f" | lift {stats['lift_over_majority_on_kept']}"
                f" | 平衡召回 {stats['balanced_recall']:.4f}"
            )
            print(
                "   分类召回: "
                + "  ".join(
                    f"{label}={value['recall']}({value['support']})"
                    for label, value in stats["per_label"].items()
                )
            )
            print(
                f"   弃答有效性 {abstention['effectiveness']}"
                f"（{abstention['sop_would_be_wrong']}/{abstention['with_sop_fallback']} 条 SOP 兜底会答错）"
            )
            print(
                f"   参照: SOP 全量准确率 {stats['sop_reference']['accuracy_over_all_cases']:.4f}"
                f" | 一律报 {stats['majority_label']} 准确率"
                f" {stats['majority_reference']['accuracy_over_all_cases']:.4f}（人工 0%）"
            )
            print(f"   M9 动作: {stats['action_breakdown']}")
            print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
