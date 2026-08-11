"""Create a reproducible split manifest for `datasets/rca_v2_l2fixed`.

This script does not rewrite case files. It only records a deterministic
train/test split and a data-quality snapshot under `_metadata/`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca_framework.anomaly import METRIC_ALIASES  # noqa: E402
from rca_framework.data import sha256_file, stratified_split_rows  # noqa: E402
from rca_framework.types import ROOT_CAUSES, SIDES  # noqa: E402


DEFAULT_DATA_DIR = Path("datasets/rca_v2_l2fixed")
MANIFEST_SCHEMA = "l2fixed-split-manifest-v1"


def _case_files(data_dir: Path) -> List[Path]:
    return sorted(data_dir.glob("case_*.json"), key=lambda path: path.name)


def _lane_width(value: Any) -> int | None:
    if isinstance(value, dict):
        return len(value)
    if isinstance(value, list):
        return len(value)
    return None


def _side_block(case: Dict[str, Any], field: str, side: str) -> Any:
    block = case.get(field)
    if isinstance(block, dict):
        return block.get(side)
    return None


def _schema_cohort(case: Dict[str, Any]) -> str:
    extended = {"task_id", "chip", "port", "fec_error", "crc_error", "port_down_dt"}
    if extended & set(case):
        return "extended"
    if {"alarm_ip_interface", "Temperature", "Voltage", "Lane number"} <= set(case):
        return "standard"
    return "partial"


def _quality_row(path: Path, case: Dict[str, Any]) -> Dict[str, Any]:
    lane_widths: Dict[str, Dict[str, int | None]] = {}
    non_numeric_blocks: List[str] = []
    missing_metrics: List[str] = []
    for metric in sorted(METRIC_ALIASES):
        source_key = next((alias for alias in METRIC_ALIASES[metric] if alias in case), metric)
        lane_widths[metric] = {}
        for side in SIDES:
            block = _side_block(case, source_key, side)
            lane_widths[metric][side] = _lane_width(block)
            if block is None:
                missing_metrics.append(f"{side}.{metric}")
            elif not isinstance(block, (dict, list)):
                non_numeric_blocks.append(f"{side}.{metric}")
    return {
        "file": path.name,
        "case_id": str(case.get("case_id", path.stem)),
        "label": str(case.get("label", "")),
        "schema_cohort": _schema_cohort(case),
        "has_alarm_ip_interface": "alarm_ip_interface" in case,
        "has_lane_number": isinstance(case.get("Lane number"), dict),
        "endpoint_values_swapped": bool(case.get("_meta", {}).get("endpoint_values_swapped"))
        if isinstance(case.get("_meta"), dict)
        else None,
        "lane_widths": lane_widths,
        "missing_metrics": missing_metrics,
        "non_numeric_blocks": non_numeric_blocks,
        "sha256": sha256_file(path),
    }


def _split_counts(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Counter[str]] = {"train": Counter(), "test": Counter()}
    for row in rows:
        counts[row["split"]][row["label"]] += 1
    return {
        split: {label: counts[split].get(label, 0) for label in ROOT_CAUSES}
        for split in ("train", "test")
    }


def _quality_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels = Counter(row["label"] for row in rows)
    cohorts = Counter(row["schema_cohort"] for row in rows)
    missing_alarm = sum(not row["has_alarm_ip_interface"] for row in rows)
    missing_lane_number = sum(not row["has_lane_number"] for row in rows)
    l1_over_four = sum(
        any((widths.get("L1") or 0) > 4 for widths in row["lane_widths"].values())
        for row in rows
    )
    l2_over_four = sum(
        any((widths.get("L2") or 0) > 4 for widths in row["lane_widths"].values())
        for row in rows
    )
    host_snr_present = sum(
        any((row["lane_widths"].get("host_snr", {}).get(side) or 0) > 0 for side in SIDES)
        for row in rows
    )
    return {
        "case_count": len(rows),
        "label_distribution": {label: labels.get(label, 0) for label in ROOT_CAUSES},
        "schema_cohorts": dict(sorted(cohorts.items())),
        "missing_alarm_ip_interface": missing_alarm,
        "missing_lane_number": missing_lane_number,
        "l1_metric_width_over_4_cases": l1_over_four,
        "l2_metric_width_over_4_cases": l2_over_four,
        "host_snr_present_cases": host_snr_present,
    }


def build_manifest(data_dir: Path, *, train_ratio: float, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    files = _case_files(data_dir)
    if not files:
        raise FileNotFoundError(f"no case_*.json files in {data_dir}")
    rows = [_quality_row(path, json.loads(path.read_text(encoding="utf-8"))) for path in files]
    unsupported = sorted({row["label"] for row in rows if row["label"] not in ROOT_CAUSES})
    if unsupported:
        raise ValueError(f"unsupported labels in {data_dir}: {unsupported}")
    split_rows = stratified_split_rows(rows, train_ratio=train_ratio, seed=seed)
    quality = _quality_summary(split_rows)
    source_payload = {
        "files": [{"file": row["file"], "sha256": row["sha256"]} for row in split_rows],
        "seed": seed,
        "train_ratio": train_ratio,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "source_dataset": data_dir.name,
        "source_dir": str(data_dir),
        "source_hash": hashlib.sha256(
            json.dumps(source_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "split_policy": {
            "name": "per-label-shuffle",
            "seed": seed,
            "train_ratio": train_ratio,
        },
        "case_count": len(split_rows),
        "split_counts": _split_counts(split_rows),
        "quality_summary": quality,
        "cases": split_rows,
    }
    return manifest, {"schema_version": "l2fixed-quality-report-v1", **quality, "cases": split_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check", action="store_true", help="verify existing metadata without writing")
    args = parser.parse_args()

    manifest, quality = build_manifest(args.data_dir, train_ratio=args.train_ratio, seed=args.seed)
    metadata = args.data_dir / "_metadata"
    manifest_path = metadata / "manifest.json"
    quality_path = metadata / "quality_report.json"
    rendered_manifest = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    rendered_quality = json.dumps(quality, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.check:
        current_manifest = manifest_path.read_text(encoding="utf-8") if manifest_path.exists() else ""
        current_quality = quality_path.read_text(encoding="utf-8") if quality_path.exists() else ""
        if current_manifest != rendered_manifest or current_quality != rendered_quality:
            raise SystemExit("l2fixed metadata is missing or stale; run scripts/prepare_l2fixed_stratified.py")
        print(f"{manifest_path} and {quality_path} are up to date")
        return
    metadata.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(rendered_manifest, encoding="utf-8")
    quality_path.write_text(rendered_quality, encoding="utf-8")
    print(f"wrote {manifest_path}")
    print(f"wrote {quality_path}")


if __name__ == "__main__":
    main()
