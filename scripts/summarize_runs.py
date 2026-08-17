#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def pct(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "-"


def load_rows(artifacts: Path) -> list[dict[str, Any]]:
    paths = sorted(artifacts.glob("**/evaluation_summary.json"))
    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        recall = summary.get("recall") or {}
        llm_modes = summary.get("llm_reasoning_mode") or {}
        status = summary.get("decision_status") or {}
        run_dir = path.parent
        rows.append(
            {
                "run": run_dir.relative_to(ROOT).as_posix() if run_dir.is_relative_to(ROOT) else run_dir.as_posix(),
                "case_count": summary.get("case_count"),
                "correct": summary.get("correct"),
                "accuracy": summary.get("accuracy"),
                "recall_L1": recall.get("L1"),
                "recall_L2": recall.get("L2"),
                "recall_fiber": recall.get("fiber"),
                "valid_llm_outputs": summary.get("valid_llm_outputs", llm_modes.get("llm_path_reasoning", 0)),
                "agreement": status.get("agreement", 0),
                "manual_review": status.get("manual_review_recommended", 0),
                "label_leakage": summary.get("label_leakage"),
            }
        )
    return rows


def print_markdown(rows: list[dict[str, Any]]) -> None:
    headers = [
        "Run",
        "Cases",
        "Correct",
        "Accuracy",
        "L1 Recall",
        "L2 Recall",
        "Fiber Recall",
        "Valid LLM",
        "Agreement",
        "Manual Review",
        "Leakage",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        print(
            "| "
            + " | ".join(
                [
                    row["run"],
                    str(row.get("case_count", "-")),
                    str(row.get("correct", "-")),
                    pct(row.get("accuracy")),
                    pct(row.get("recall_L1")),
                    pct(row.get("recall_L2")),
                    pct(row.get("recall_fiber")),
                    str(row.get("valid_llm_outputs", "-")),
                    str(row.get("agreement", "-")),
                    str(row.get("manual_review", "-")),
                    str(row.get("label_leakage", "-")),
                ]
            )
            + " |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize RCA v2 evaluation_summary.json files.")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of Markdown.")
    args = parser.parse_args()

    artifacts = args.artifacts if args.artifacts.is_absolute() else ROOT / args.artifacts
    rows = load_rows(artifacts)
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        if not rows:
            print(f"No evaluation_summary.json files found under {artifacts}")
            return
        print_markdown(rows)


if __name__ == "__main__":
    main()
