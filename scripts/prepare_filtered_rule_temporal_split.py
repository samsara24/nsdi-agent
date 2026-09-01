#!/usr/bin/env python3
"""Freeze the filtered-rule datasets with a 2025-06..09 temporal train split.

The two source datasets share one materialized training directory.  Their
source-relative endpoint aliases are normalized to the common L1(local) /
L2(remote) / fiber label space.  Cases outside the training window remain in
source-specific test directories because their lane profiles are different.
Source files are never changed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rca_framework.topology import SOURCE_TOPOLOGIES, TOPOLOGY_CONTRACT_VERSION  # noqa: E402
DEFAULT_SOURCES = {
    "all_data": ROOT / "datasets/filtered_by_rule/all_data",
    "rule1_channel_not_4": ROOT / "datasets/filtered_by_rule/rule1_channel_not_4",
}
DEFAULT_OUTPUT = ROOT / "datasets/filtered_rule_temporal_2025_06_09_v1"
DEFAULT_ANNOTATIONS = Path("/Users/ziangchen/Downloads/expert_label_annotations.json")
DEFAULT_REFERENCES = (
    ROOT / "experiments/20260816_expanded-pattern-conflict/clean_train.jsonl",
    ROOT / "experiments/20260816_expanded-pattern-conflict/clean_expanded_test.jsonl",
)
TRAIN_MONTHS = ("2025-06", "2025-07", "2025-08", "2025-09")
MEASUREMENT_KEYS = (
    "bias", "rxpower", "txpower", "media_snr", "host_snr", "serdes_snr",
    "RxLOL", "TxLOL", "TxLOS", "RxLOS", "Temperature", "Voltage",
)
VALID_LABELS = {
    "all_data": {"l1", "l2", "fiber"},
    "rule1_channel_not_4": {"l3", "l4", "fiber"},
}
SOURCE_ENDPOINT_ALIASES = {
    "all_data": {"l1": "L1", "l2": "L2"},
    "rule1_channel_not_4": {"l3": "L1", "l4": "L2"},
}
UNIFIED_LABELS = {"L1", "L2", "fiber"}


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def parse_alarm_time(value: Any) -> datetime:
    text = " ".join(str(value or "").strip().split())
    for pattern in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M"):
        try:
            return datetime.strptime(text, pattern)
        except ValueError:
            continue
    raise ValueError(f"unsupported alarm_time: {value!r}")


def normalized(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key).lower(): normalized(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]).lower())
        }
    if isinstance(value, list):
        return [normalized(item) for item in value]
    return value


def canonical_endpoint_key(key: str, aliases: dict[str, str]) -> str:
    """Normalize exact endpoint tokens, including directional keys such as l3-l4."""
    return "-".join(aliases.get(token, token) for token in key.split("-"))


def canonicalize_endpoints(value: Any, aliases: dict[str, str]) -> Any:
    """Recursively normalize endpoint-bearing dictionary keys without changing values."""
    if isinstance(value, dict):
        return {
            canonical_endpoint_key(str(key), aliases): canonicalize_endpoints(item, aliases)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [canonicalize_endpoints(item, aliases) for item in value]
    return copy.deepcopy(value)


def unified_label(source: str, source_label: str) -> str:
    if source_label == "fiber":
        return "fiber"
    try:
        return SOURCE_ENDPOINT_ALIASES[source][source_label]
    except KeyError as exc:
        raise ValueError(f"cannot map label {source_label!r} from {source!r}") from exc


def measurement_fingerprint(case: dict[str, Any]) -> str:
    """Match old/new cases using exact telemetry while ignoring labels and metadata."""
    payload = {key: normalized(case.get(key)) for key in MEASUREMENT_KEYS}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def stable_case_id(source: str, case: dict[str, Any]) -> str:
    payload = {key: value for key, value in case.items() if key not in {"label", "case_id", "_dataset_contract"}}
    encoded = json.dumps(
        {"source": source, "case": payload}, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return f"case_{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:16]}"


def load_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def expert_proposals(annotation_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "rca-expert-label-review-v1":
        raise ValueError(f"unsupported annotation schema: {payload.get('schema_version')!r}")
    by_case: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"explicit_labels": set(), "proposals": set(), "annotations": []}
    )
    completed = 0
    for annotation in payload.get("annotations", []):
        if not annotation.get("completed"):
            continue
        completed += 1
        for side in ("left", "right"):
            case_id = str(annotation[f"{side}_case_id"])
            proposal = str(annotation.get(f"{side}_label", "keep"))
            if proposal not in {"keep", "uncertain", "L1", "L2", "fiber"}:
                raise ValueError(f"unsupported expert proposal {proposal!r} for {case_id}")
            row = by_case[case_id]
            row["proposals"].add(proposal)
            if proposal in UNIFIED_LABELS:
                row["explicit_labels"].add(proposal)
            row["annotations"].append({
                "pattern_id": annotation.get("pattern_id"),
                "decision": annotation.get("decision"),
                "evidence_status": annotation.get("evidence_status"),
                "notes": annotation.get("notes"),
                "updated_at": annotation.get("updated_at"),
            })
    result: dict[str, dict[str, Any]] = {}
    for case_id, row in by_case.items():
        if len(row["explicit_labels"]) > 1:
            raise ValueError(f"conflicting expert labels for {case_id}: {sorted(row['explicit_labels'])}")
        result[case_id] = {
            "explicit_label": next(iter(row["explicit_labels"]), None),
            "proposals": sorted(row["proposals"]),
            "annotations": row["annotations"],
        }
    summary = {
        "schema_version": payload["schema_version"],
        "exported_at": payload.get("exported_at"),
        "annotation_count": len(payload.get("annotations", [])),
        "completed_annotation_count": completed,
        "reviewed_case_id_count": len(result),
    }
    return result, summary


def build_expert_fingerprint_index(
    reference_cases: Iterable[dict[str, Any]], proposals: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    for case in reference_cases:
        case_id = str(case.get("case_id", ""))
        if case_id in proposals:
            index[measurement_fingerprint(case)].add(case_id)
    return {fingerprint: sorted(case_ids) for fingerprint, case_ids in index.items()}


def lane_widths(case: dict[str, Any]) -> dict[str, list[int]]:
    result: dict[str, set[int]] = defaultdict(set)
    for metric in ("bias", "rxpower", "txpower", "media_snr", "host_snr", "serdes_snr"):
        value = case.get(metric)
        if not isinstance(value, dict):
            continue
        for side, lanes in value.items():
            if isinstance(lanes, dict):
                result[str(side)].add(len(lanes))
    return {side: sorted(widths) for side, widths in sorted(result.items())}


def source_snapshot(source_root: Path) -> tuple[list[Path], str]:
    paths = sorted(source_root.rglob("*.json"), key=lambda path: path.relative_to(source_root).as_posix())
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(source_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return paths, digest.hexdigest()


def prepare(
    sources: dict[str, Path], output_dir: Path, annotation_path: Path, reference_paths: tuple[Path, ...],
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty dataset: {output_dir}")
    proposals, annotation_summary = expert_proposals(annotation_path)
    references = load_jsonl(reference_paths)
    expert_index = build_expert_fingerprint_index(references, proposals)

    records: list[dict[str, Any]] = []
    label_changes: list[dict[str, Any]] = []
    date_repairs: list[dict[str, Any]] = []
    source_summaries: dict[str, Any] = {}
    used_expert_ids: set[str] = set()
    seen_case_ids: set[str] = set()

    for source, source_root in sources.items():
        paths, snapshot_hash = source_snapshot(source_root)
        month_counts: Counter[str] = Counter()
        original_labels: Counter[str] = Counter()
        final_labels: Counter[str] = Counter()
        split_counts: Counter[str] = Counter()
        measured_widths: Counter[str] = Counter()
        source_records: list[dict[str, Any]] = []
        for path in paths:
            case = json.loads(path.read_text(encoding="utf-8"))
            original_label = str(case.get("label"))
            if original_label not in VALID_LABELS[source]:
                raise ValueError(f"unexpected label {original_label!r} in {path}")
            parsed_time = parse_alarm_time(case.get("alarm_time"))
            month = parsed_time.strftime("%Y-%m")
            split = "train" if month in TRAIN_MONTHS else f"test/{source}"
            case_id = stable_case_id(source, case)
            if case_id in seen_case_ids:
                raise ValueError(f"duplicate stable case id: {case_id}")
            seen_case_ids.add(case_id)

            expert_case_ids = expert_index.get(measurement_fingerprint(case), [])
            explicit_labels = {
                proposals[expert_id]["explicit_label"]
                for expert_id in expert_case_ids
                if proposals[expert_id]["explicit_label"] is not None
            }
            if len(explicit_labels) > 1:
                raise ValueError(f"ambiguous expert mapping for {path}: {sorted(explicit_labels)}")
            source_adjusted_label = original_label
            label_status = "unreviewed"
            if expert_case_ids:
                if source != "all_data":
                    raise ValueError(f"expert L1/L2 annotation cannot be mapped safely to {source}: {path}")
                used_expert_ids.update(expert_case_ids)
                label_status = "expert_reviewed"
                if any("uncertain" in proposals[expert_id]["proposals"] for expert_id in expert_case_ids):
                    label_status = "expert_reviewed_uncertain"
                if explicit_labels:
                    source_adjusted_label = next(iter(explicit_labels)).lower()
            final_label = unified_label(source, source_adjusted_label)
            output_case = canonicalize_endpoints(case, SOURCE_ENDPOINT_ALIASES[source])
            output_case["case_id"] = case_id
            output_case["label"] = final_label
            output_case["_dataset_contract"] = {
                "schema_version": "filtered-rule-temporal-split-v1",
                "source_dataset": source,
                "source_file": path.relative_to(source_root).as_posix(),
                "source_sha256": sha256_file(path),
                "split": split,
                "parsed_month": month,
                "original_alarm_time": case.get("alarm_time"),
                "original_label": original_label,
                "source_adjusted_label": source_adjusted_label,
                "unified_label": final_label,
                "endpoint_aliases": SOURCE_ENDPOINT_ALIASES[source],
                "topology_contract_version": TOPOLOGY_CONTRACT_VERSION,
                "topology_id": SOURCE_TOPOLOGIES[source]["topology_id"],
                "endpoint_speeds": SOURCE_TOPOLOGIES[source]["endpoint_speeds"],
                "lane_alignment": {
                    "same_index_optical_pairing": True,
                    "basis": "source transmission field is computed from same-index far RX minus near TX for non-sentinel readings",
                    "absolute_link_loss_allowed": False,
                    "serdes_to_optical_pairing_allowed": False,
                },
                "label_status": label_status,
                "expert_reference_case_ids": expert_case_ids,
                "measurement_lane_widths": lane_widths(case),
            }
            output_path = output_dir / split / f"{case_id}.json"
            dump_json(output_path, output_case)
            output_sha = sha256_file(output_path)
            row = {
                "case_id": case_id,
                "source_dataset": source,
                "source_file": path.relative_to(source_root).as_posix(),
                "source_sha256": sha256_file(path),
                "output_file": output_path.relative_to(output_dir).as_posix(),
                "output_sha256": output_sha,
                "split": split,
                "month": month,
                "original_label": original_label,
                "source_adjusted_label": source_adjusted_label,
                "label": final_label,
                "label_status": label_status,
                "expert_reference_case_ids": expert_case_ids,
            }
            records.append(row)
            source_records.append(row)
            month_counts[month] += 1
            original_labels[original_label] += 1
            final_labels[final_label] += 1
            split_counts[split] += 1
            for side, widths in lane_widths(case).items():
                for width in widths:
                    measured_widths[f"{side}:{width}"] += 1
            if "/" in str(case.get("alarm_time", "")):
                date_repairs.append({
                    "case_id": case_id, "source_dataset": source,
                    "source_file": path.relative_to(source_root).as_posix(),
                    "original_alarm_time": case.get("alarm_time"), "parsed_month": month,
                })
            if source_adjusted_label != original_label:
                label_changes.append({
                    "case_id": case_id, "source_dataset": source,
                    "source_file": path.relative_to(source_root).as_posix(), "split": split,
                    "original_label": original_label,
                    "source_adjusted_label": source_adjusted_label,
                    "unified_label": final_label,
                    "expert_reference_case_ids": expert_case_ids,
                    "expert_annotations": [
                        item
                        for expert_id in expert_case_ids
                        for item in proposals[expert_id]["annotations"]
                    ],
                })
        source_summaries[source] = {
            "source_path": portable_path(source_root),
            "source_snapshot_sha256": snapshot_hash,
            "case_count": len(source_records),
            "month_counts": dict(sorted(month_counts.items())),
            "split_counts": dict(sorted(split_counts.items())),
            "original_label_distribution": dict(sorted(original_labels.items())),
            "unified_label_distribution": dict(sorted(final_labels.items())),
            "endpoint_aliases": SOURCE_ENDPOINT_ALIASES[source],
            "measurement_lane_width_observations": dict(sorted(measured_widths.items())),
            "expert_reviewed_match_count": sum(row["label_status"] != "unreviewed" for row in source_records),
            "expert_label_change_count": sum(
                row["original_label"] != row["source_adjusted_label"] for row in source_records
            ),
        }

    records.sort(key=lambda row: (row["split"], row["source_dataset"], row["month"], row["case_id"]))
    split_counts = Counter(row["split"] for row in records)
    split_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in records:
        split_label_counts[row["split"]][row["label"]] += 1
    manifest = {
        "schema_version": "filtered-rule-temporal-split-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "train": "Cases whose parsed alarm month is 2025-06 through 2025-09 inclusive, combined across both sources.",
            "test": "All remaining months, kept in two source-specific test directories because endpoint naming and lane profiles differ.",
            "labels": "Normalize l1/l3 to L1 (local), l2/l4 to L2 (remote), and preserve fiber; exact expert annotations are applied before normalization.",
            "experiments": "No model fitting or evaluation is performed by this preparation script.",
        },
        "train_months": list(TRAIN_MONTHS),
        "unified_label_space": {
            "labels": ["L1", "L2", "fiber"],
            "semantics": {"L1": "local endpoint", "L2": "remote endpoint", "fiber": "link medium"},
            "source_endpoint_aliases": SOURCE_ENDPOINT_ALIASES,
        },
        "topology_contract": {
            "version": TOPOLOGY_CONTRACT_VERSION,
            "sources": SOURCE_TOPOLOGIES,
            "policy": (
                "Same-index optical lanes are logical pairs defined by the source telemetry contract. "
                "They may be used for categorical and within-case relative evidence, never for absolute link loss. "
                "SerDes lanes are not paired to optical lanes."
            ),
        },
        "case_count": len(records),
        "split_counts": dict(sorted(split_counts.items())),
        "split_label_distributions": {
            split: dict(sorted(counts.items())) for split, counts in sorted(split_label_counts.items())
        },
        "sources": source_summaries,
        "expert_annotations": {
            **annotation_summary,
            "source_path": portable_path(annotation_path),
            "source_sha256": sha256_file(annotation_path),
            "reference_paths": [portable_path(path) for path in reference_paths],
            "reference_sha256": {portable_path(path): sha256_file(path) for path in reference_paths},
            "matched_reviewed_case_count": sum(row["label_status"] != "unreviewed" for row in records),
            "changed_case_count": len(label_changes),
            "unmatched_reviewed_case_ids": sorted(set(proposals) - used_expert_ids),
        },
        "date_repair_count": len(date_repairs),
        "cases": records,
    }
    expected_from_image = {
        "all_data": {"total": 507, "2025-06": 3, "2025-07": 2, "2025-08": 34, "2025-09": 50},
        "rule1_channel_not_4": {"total": 109, "2025-06": 5, "2025-07": 14, "2025-08": 14, "2025-09": 5},
    }
    image_differences = {}
    for source, expected in expected_from_image.items():
        actual_months = source_summaries[source]["month_counts"]
        actual = {"total": source_summaries[source]["case_count"], **{m: actual_months.get(m, 0) for m in TRAIN_MONTHS}}
        image_differences[source] = {
            "expected_from_image": expected,
            "actual_files": actual,
            "delta_actual_minus_image": {key: actual[key] - expected[key] for key in expected},
        }
    quality_report = {
        "schema_version": "filtered-rule-temporal-quality-v1",
        "confirmed_source_contract": {
            "confirmed_at": "2026-08-22",
            "all_data": {"case_count": 505, "label_distribution": {"l1": 179, "l2": 305, "fiber": 21}},
            "rule1_channel_not_4": {"case_count": 103, "label_distribution": {"l3": 49, "l4": 50, "fiber": 4}},
            "policy": "The source files and their observed label distributions are authoritative; screenshot counts are non-contractual reference only.",
        },
        "source_count_difference_from_user_image": image_differences,
        "date_repairs": date_repairs,
        "label_change_count": len(label_changes),
        "expert_reviewed_match_count": sum(row["label_status"] != "unreviewed" for row in records),
        "unmatched_expert_case_id_count": len(set(proposals) - used_expert_ids),
        "notes": [
            "The user confirmed that the source directories' 505/103 JSON files are complete; differing screenshot counts are non-contractual and do not indicate missing cases.",
            "Lane number is absent in rule1_channel_not_4, so lane profiles are audited from measurement-array widths.",
            "The user confirmed a shared endpoint-relative label space: l1/l3 map to L1 (local), l2/l4 map to L2 (remote), and fiber remains fiber.",
            "Directory names and alarm-side fields are not used to infer labels; normalization uses only the source-specific endpoint aliases.",
        ],
    }
    dump_json(output_dir / "_metadata/manifest.json", manifest)
    dump_json(output_dir / "_metadata/quality_report.json", quality_report)
    dump_json(output_dir / "_metadata/label_audit.json", {
        "schema_version": "filtered-rule-label-audit-v1",
        "expert_source_sha256": sha256_file(annotation_path),
        "reviewed_matches": [row for row in records if row["label_status"] != "unreviewed"],
        "label_changes": label_changes,
        "unmatched_reviewed_case_ids": sorted(set(proposals) - used_expert_ids),
    })
    (output_dir / "_metadata/expert_label_annotations.source.json").write_bytes(annotation_path.read_bytes())
    return manifest


def check(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "_metadata/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for row in manifest.get("cases", []):
        path = output_dir / row["output_file"]
        if not path.exists():
            errors.append(f"missing output: {row['output_file']}")
            continue
        if sha256_file(path) != row["output_sha256"]:
            errors.append(f"output hash mismatch: {row['output_file']}")
        case = json.loads(path.read_text(encoding="utf-8"))
        if case.get("case_id") != row["case_id"] or case.get("label") != row["label"]:
            errors.append(f"case contract mismatch: {row['output_file']}")
        source_root = Path(manifest["sources"][row["source_dataset"]]["source_path"])
        if not source_root.is_absolute():
            source_root = ROOT / source_root
        source_path = source_root / row["source_file"]
        if not source_path.exists() or sha256_file(source_path) != row["source_sha256"]:
            errors.append(f"source drift: {row['source_dataset']}/{row['source_file']}")
    actual_json = sorted(
        path for path in output_dir.rglob("*.json")
        if "_metadata" not in path.parts
    )
    if len(actual_json) != manifest.get("case_count"):
        errors.append(f"case count mismatch: manifest={manifest.get('case_count')} files={len(actual_json)}")
    result = {"ok": not errors, "case_count": len(actual_json), "errors": errors}
    if errors:
        raise ValueError(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-data", type=Path, default=DEFAULT_SOURCES["all_data"])
    parser.add_argument("--rule1", type=Path, default=DEFAULT_SOURCES["rule1_channel_not_4"])
    parser.add_argument("--expert-annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        result = check(args.output_dir)
    else:
        manifest = prepare(
            {"all_data": args.all_data, "rule1_channel_not_4": args.rule1},
            args.output_dir, args.expert_annotations, DEFAULT_REFERENCES,
        )
        result = {
            "output_dir": portable_path(args.output_dir),
            "schema_version": manifest["schema_version"],
            "case_count": manifest["case_count"],
            "split_counts": manifest["split_counts"],
            "expert_reviewed_match_count": manifest["expert_annotations"]["matched_reviewed_case_count"],
            "expert_label_change_count": manifest["expert_annotations"]["changed_case_count"],
            "date_repair_count": manifest["date_repair_count"],
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
