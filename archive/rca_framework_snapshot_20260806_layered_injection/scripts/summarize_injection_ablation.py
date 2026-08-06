#!/usr/bin/env python
"""Summarize KG-injection ablation predictions into a compact analysis JSON."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ARMS = (
    "full__legacy",
    "full__llm_only",
    "layered__legacy",
    "layered__llm_only",
)


def load_rows(root: Path, arm: str) -> list[dict[str, Any]]:
    path = root / arm / "predictions.json"
    return json.loads(path.read_text(encoding="utf-8"))


def accuracy(rows: list[dict[str, Any]]) -> float | None:
    return sum(bool(row["correct"]) for row in rows) / len(rows) if rows else None


def summarize_arm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_regime: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_sufficiency: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_regime_actual: Counter[tuple[str, str]] = Counter()
    sufficiency_cross: Counter[tuple[str, str, str]] = Counter()
    prompt_lengths: dict[str, list[int]] = defaultdict(list)
    zero_anomaly_by_regime: Counter[str] = Counter()

    for row in rows:
        llm = row["KG_RAG_LLM"]
        regime = str(llm["kg_coverage"]["regime"])
        sufficiency = str(llm.get("evidence_sufficiency", "unreported"))
        actual = str(row["actual_label"])
        by_regime[regime].append(row)
        by_sufficiency[sufficiency].append(row)
        by_regime_actual[(regime, actual)] += 1
        sufficiency_cross[(sufficiency, regime, actual)] += 1
        prompt_lengths[regime].append(len(llm.get("prompt", "")))
        if not row.get("extracted_anomalies"):
            zero_anomaly_by_regime[regime] += 1

    return {
        "case_count": len(rows),
        "correct": sum(bool(row["correct"]) for row in rows),
        "accuracy": accuracy(rows),
        "final_prediction_distribution": dict(Counter(row["prediction"] for row in rows)),
        "llm_prediction_distribution": dict(Counter(row["KG_RAG_LLM"]["prediction"] for row in rows)),
        "llm_route_accuracy": sum(
            row["KG_RAG_LLM"]["prediction"] == row["actual_label"] for row in rows
        ) / len(rows),
        "symbolic_prediction_distribution": dict(Counter(row["KG_RCA"]["prediction"] for row in rows)),
        "regime": {
            regime: {
                "cases": len(group),
                "correct": sum(bool(row["correct"]) for row in group),
                "accuracy": accuracy(group),
                "actual_distribution": dict(
                    Counter(row["actual_label"] for row in group)
                ),
                "final_prediction_distribution": dict(
                    Counter(row["prediction"] for row in group)
                ),
                "llm_prediction_distribution": dict(
                    Counter(row["KG_RAG_LLM"]["prediction"] for row in group)
                ),
                "mean_prompt_characters": round(mean(prompt_lengths[regime]), 1),
                "zero_anomaly_cases": zero_anomaly_by_regime[regime],
            }
            for regime, group in sorted(by_regime.items())
        },
        "evidence_sufficiency": {
            value: {
                "cases": len(group),
                "correct": sum(bool(row["correct"]) for row in group),
                "accuracy": accuracy(group),
                "actual_distribution": dict(
                    Counter(row["actual_label"] for row in group)
                ),
                "regime_distribution": dict(
                    Counter(row["KG_RAG_LLM"]["kg_coverage"]["regime"] for row in group)
                ),
            }
            for value, group in sorted(by_sufficiency.items())
        },
        "sufficiency_regime_actual": {
            f"{sufficiency}/{regime}/{actual}": count
            for (sufficiency, regime, actual), count in sorted(sufficiency_cross.items())
        },
    }


def compare(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    left_name: str,
    right_name: str,
) -> dict[str, Any]:
    left = {row["case_id"]: row for row in left_rows}
    right = {row["case_id"]: row for row in right_rows}
    changed = []
    improved = 0
    worsened = 0
    for case_id in sorted(left):
        old, new = left[case_id], right[case_id]
        if old["prediction"] == new["prediction"]:
            continue
        if not old["correct"] and new["correct"]:
            effect = "improved"
            improved += 1
        elif old["correct"] and not new["correct"]:
            effect = "worsened"
            worsened += 1
        else:
            effect = "changed_but_still_wrong"
        changed.append({
            "case_id": case_id,
            "actual": old["actual_label"],
            "regime": new["KG_RAG_LLM"]["kg_coverage"]["regime"],
            "from": old["prediction"],
            "to": new["prediction"],
            "effect": effect,
            "left_llm_prediction": old["KG_RAG_LLM"]["prediction"],
            "right_llm_prediction": new["KG_RAG_LLM"]["prediction"],
            "right_sufficiency": new["KG_RAG_LLM"].get("evidence_sufficiency"),
        })
    return {
        "left": left_name,
        "right": right_name,
        "final_prediction_changes": len(changed),
        "improved": improved,
        "worsened": worsened,
        "net_correct_change": improved - worsened,
        "llm_prediction_changes": sum(
            left[case_id]["KG_RAG_LLM"]["prediction"]
            != right[case_id]["KG_RAG_LLM"]["prediction"]
            for case_id in left
        ),
        "raw_output_changes": sum(
            left[case_id]["KG_RAG_LLM"].get("raw_output")
            != right[case_id]["KG_RAG_LLM"].get("raw_output")
            for case_id in left
        ),
        "changed_cases": changed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = {arm: load_rows(args.run_dir, arm) for arm in ARMS}
    result = {
        "arms": {arm: summarize_arm(items) for arm, items in rows.items()},
        "comparisons": {
            "proposed_vs_original": compare(
                rows["full__legacy"], rows["layered__llm_only"],
                "full__legacy", "layered__llm_only",
            ),
            "layering_only": compare(
                rows["full__legacy"], rows["layered__legacy"],
                "full__legacy", "layered__legacy",
            ),
            "score_only_full_prompt": compare(
                rows["full__legacy"], rows["full__llm_only"],
                "full__legacy", "full__llm_only",
            ),
            "score_only_layered_prompt": compare(
                rows["layered__legacy"], rows["layered__llm_only"],
                "layered__legacy", "layered__llm_only",
            ),
        },
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
