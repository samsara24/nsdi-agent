#!/usr/bin/env python3
"""Materialize a legacy positional dataset from JSONL train/test contracts.

The old positional prefix remains available for historical reproduction.  New
expanded expert-clean runs pass ``--train-jsonl`` so removed cases and adjudicated
training labels are reproduced exactly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-data-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=126)
    parser.add_argument("--train-jsonl", type=Path)
    parser.add_argument("--expanded-test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    old_files = sorted(args.old_data_dir.glob("case_*.json"))
    if args.train_jsonl is None and len(old_files) < args.train_size:
        raise ValueError(f"old dataset has only {len(old_files)} cases")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {args.output_dir}")

    tests = [json.loads(line) for line in args.expanded_test.read_text(encoding="utf-8").splitlines() if line.strip()]
    trains = (
        [json.loads(line) for line in args.train_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        if args.train_jsonl is not None
        else [json.loads(path.read_text(encoding="utf-8")) for path in old_files[: args.train_size]]
    )
    if len(trains) != args.train_size:
        raise ValueError(f"train size mismatch: expected {args.train_size}, got {len(trains)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_ids = []
    for index, case in enumerate(trains, 1):
        train_ids.append(str(case.get("case_id", f"train_{index}")))
        (args.output_dir / f"case_{index:06d}.json").write_text(
            json.dumps(case, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    test_ids = []
    for offset, case in enumerate(tests, args.train_size + 1):
        test_ids.append(str(case.get("case_id", f"expanded_{offset}")))
        (args.output_dir / f"case_{offset:06d}.json").write_text(
            json.dumps(case, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    metadata = args.output_dir / "_metadata"
    metadata.mkdir()
    manifest = {
        "schema_version": "expanded-legacy-positional-v2",
        "train_size": len(train_ids),
        "test_size": len(test_ids),
        "train_case_ids": train_ids,
        "test_case_ids": test_ids,
        "train_source": str(args.train_jsonl) if args.train_jsonl is not None else f"first {args.train_size} cases of {args.old_data_dir}",
        "test_source": str(args.expanded_test),
        "policy": "Versioned train/test JSONL contract materialized positionally; labels are retained for fitting/evaluation only and removed from inference evidence.",
    }
    (metadata / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "train_size": len(train_ids), "test_size": len(test_ids)}))


if __name__ == "__main__":
    main()
