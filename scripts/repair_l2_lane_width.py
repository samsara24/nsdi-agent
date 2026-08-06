#!/usr/bin/env python3
"""Create a copy of the v2 dataset with invalid 200G-side lane vectors repaired.

The source data contains a small group of 400G--200G alarm cases whose L2
vectors have 7 or 8 entries, while normal 200G cases have four.  The first
four entries are retained consistently across all side-scoped lane metrics.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


LANE_FIELDS = ("bias", "rxpower", "txpower", "media_snr", "host_snr", "serdes_snr")
EXPECTED_LANES = ("0", "1", "2", "3")


def repair_case(case: dict) -> tuple[dict, list[dict]]:
    changes: list[dict] = []
    for field in LANE_FIELDS:
        block = case.get(field)
        if not isinstance(block, dict) or not isinstance(block.get("L2"), dict):
            continue
        values = block["L2"]
        keys = list(values)
        numeric_keys = sorted((key for key in keys if str(key).isdigit()), key=lambda key: int(key))
        if len(numeric_keys) <= len(EXPECTED_LANES):
            continue
        kept = {key: values[key] for key in EXPECTED_LANES if key in values}
        block["L2"] = kept
        changes.append({"field": field, "before": len(numeric_keys), "after": len(kept)})
    if changes:
        case.setdefault("_meta", {})["l2_lane_repair"] = {
            "rule": "retain L2 lane keys 0-3; remove excess keys",
            "changed_fields": changes,
        }
    return case, changes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)

    report = {"source": str(args.source), "cases": 0, "repaired_cases": 0, "changed_fields": 0, "details": []}
    for source_file in sorted(args.source.glob("case_*.json")):
        case = json.loads(source_file.read_text(encoding="utf-8"))
        case, changes = repair_case(case)
        target_file = args.output / source_file.name
        target_file.write_text(json.dumps(case, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report["cases"] += 1
        if changes:
            report["repaired_cases"] += 1
            report["changed_fields"] += len(changes)
            report["details"].append({"case_id": case.get("case_id"), "file": source_file.name, "changes": changes})

    metadata = args.output / "_metadata"
    metadata.mkdir()
    (metadata / "l2_lane_repair_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in ("cases", "repaired_cases", "changed_fields")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
