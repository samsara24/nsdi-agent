#!/usr/bin/env python3
"""Export raw telemetry packets for independent model review, with labels removed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FORBIDDEN_KEYS = {
    "label", "original_label", "unified_label", "source_adjusted_label",
    "label_status", "expert_reference_case_ids", "source_file", "source_sha256",
    "_dataset_contract",
}


def scrub(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: scrub(item)
            for key, item in value.items()
            if key not in FORBIDDEN_KEYS and "label" not in key.lower()
        }
    if isinstance(value, list):
        return [scrub(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/filtered_rule_temporal_2025_06_09_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/current_model_case_review_v1/blind_packets"))
    args = parser.parse_args()
    for split in ("all_data", "rule1_channel_not_4"):
        rows = []
        for path in sorted((args.data_dir / "test" / split).glob("*.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            packet = scrub(raw)
            if any("label" in key.lower() for key in packet):
                raise RuntimeError(f"label-like key survived scrub: {path}")
            rows.append(packet)
        target = args.output_dir / f"{split}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(split, len(rows), target)


if __name__ == "__main__":
    main()
