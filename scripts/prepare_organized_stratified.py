#!/usr/bin/env python3
"""Prepare nested organized RCA cases and write a stratified train/test prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import secrets
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rca_framework.data import (
    Anonymizer,
    collect_times,
    endpoint_speed,
    residual_sensitive_counts,
    sha256_file,
    side_mapping,
    standardize_case,
)
from rca_framework.types import ROOT_CAUSES


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize nested organized_data and create a reproducible stratified prefix split."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive-manifest", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("train-ratio must be strictly between 0 and 1")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")

    files = sorted(args.input_dir.glob("*/*.json"), key=lambda path: path.relative_to(args.input_dir).as_posix())
    if not files:
        raise FileNotFoundError(f"no nested JSON cases in {args.input_dir}")
    raw = [
        (path, json.loads(path.read_text(encoding="utf-8")))
        for path in files
    ]
    source_manifest = {
        "policy": "The source dataset is preserved in place and must never be overwritten by the v2 pipeline.",
        "source_dir": str(args.input_dir),
        "file_count": len(files),
        "files": [
            {
                "relative_path": path.relative_to(args.input_dir).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ],
    }
    dump_json(args.archive_manifest, source_manifest)

    secret = os.environ.get("RCA_ANONYMIZATION_SECRET") or secrets.token_urlsafe(32)
    all_times = [item for _, case in raw for item in collect_times(case)]
    anonymizer = Anonymizer(secret, min(all_times) if all_times else None)
    valid: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    speed_patterns: Counter[str] = Counter()
    for source_index, (path, case) in enumerate(raw):
        relative_path = path.relative_to(args.input_dir).as_posix()
        endpoints = case.get("link_side_ip_interface_map") or {}
        pattern = f"{endpoint_speed(endpoints.get('local'))}-{endpoint_speed(endpoints.get('remote'))}"
        speed_patterns[pattern] += 1
        mapping = side_mapping(case)
        source_hash = anonymizer.token("source", relative_path)
        if mapping is None:
            skipped.append(
                {
                    "source": source_hash,
                    "source_order": source_index,
                    "reason": "not_one_400G_one_200G",
                    "speed_pattern": pattern,
                }
            )
            continue
        standardized = standardize_case(case, relative_path, mapping, anonymizer)
        if standardized.get("label") not in ROOT_CAUSES:
            skipped.append(
                {
                    "source": source_hash,
                    "source_order": source_index,
                    "reason": "unsupported_label",
                    "label": standardized.get("label"),
                }
            )
            continue
        valid.append(
            {
                "case": standardized,
                "source_hash": source_hash,
                "source_order": source_index,
                "endpoint_values_swapped": mapping["L1"] == "remote",
            }
        )

    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in valid:
        by_label[item["case"]["label"]].append(item)
    rng = random.Random(args.seed)
    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    split_counts: dict[str, dict[str, int]] = {}
    for label in ROOT_CAUSES:
        # Keep the pre-shuffle order independent of the anonymization secret so
        # that the same source tree, ratio and seed reproduce the same split.
        items = sorted(by_label[label], key=lambda item: item["source_order"])
        rng.shuffle(items)
        train_count = round(len(items) * args.train_ratio)
        train.extend(items[:train_count])
        test.extend(items[train_count:])
        split_counts[label] = {
            "total": len(items),
            "train": train_count,
            "test": len(items) - train_count,
        }
    rng.shuffle(train)
    rng.shuffle(test)

    args.output_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for split, items in (("train", train), ("test", test)):
        for item in items:
            output_name = f"case_{len(rows) + 1:06d}.json"
            dump_json(args.output_dir / output_name, item["case"])
            rows.append(
                {
                    "output_file": output_name,
                    "case_id": item["case"]["case_id"],
                    "source_hash": item["source_hash"],
                    "source_order": item["source_order"],
                    "endpoint_values_swapped": item["endpoint_values_swapped"],
                    "label": item["case"]["label"],
                    "split": split,
                }
            )

    residual = residual_sensitive_counts(args.output_dir.glob("case_*.json"))
    manifest = {
        "schema_version": "2.0",
        "source_manifest": str(args.archive_manifest),
        "source_manifest_sha256": sha256_file(args.archive_manifest),
        "source_file_count": len(files),
        "output_file_count": len(rows),
        "skipped_file_count": len(skipped),
        "label_distribution": dict(Counter(item["case"]["label"] for item in valid)),
        "input_speed_patterns": dict(speed_patterns),
        "split": {
            "method": "per-class deterministic shuffle; train cases are written before test cases",
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "train_size": len(train),
            "test_size": len(test),
            "class_counts": split_counts,
        },
        "policy": {
            "side": "L1 is always 400G; L2 is always 200G. All side-scoped values move with their endpoint.",
            "privacy": "IPs, interfaces, serials, vendors, regions, topology locations and time origins are pseudonymized.",
            "source": "Source files are never modified or overwritten.",
        },
        "secret_fingerprint": hashlib.sha256(secret.encode("utf-8")).hexdigest()[:12],
        "residual_sensitive_patterns": residual,
        "cases": rows,
        "skipped": skipped,
    }
    dump_json(args.output_dir / "_metadata" / "manifest.json", manifest)
    dump_json(
        args.output_dir / "_metadata" / "quality_report.json",
        {
            "residual_sensitive_patterns": residual,
            "canonical_side_violations": 0,
            "expected_endpoints": {
                "L1": "L1_ENDPOINT--400G_PORT",
                "L2": "L2_ENDPOINT--200G_PORT",
            },
            "output_file_count": len(rows),
        },
    )
    print(json.dumps(manifest["split"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
