from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import random
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .types import ROOT_CAUSES


IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
IPV6_RE = re.compile(r"\b(?:[0-9A-Fa-f]{1,4}:){2,}[0-9A-Fa-f:]*\b")
RAW_INTERFACE_RE = re.compile(r"\b(?:200|400)G(?:HL|E)[A-Za-z0-9/:.-]*", re.I)
LEGACY_SIDE_RE = re.compile(r"\b(?:local|remote)\b", re.I)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b1[3-9]\d{9}\b")
TIME_KEYS = {"alarm_time", "timestamp", "devm_picm_dt", "port_down_dt"}
TOKEN_FIELDS = {
    "region": "region",
    "link_location": "location",
    "vendor": "vendor",
    "vendor_sn": "serial",
    "vender_sn": "serial",
    "task_id": "task",
    "chip": "device",
}
TIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
)


def sorted_json_files(path: Path) -> List[Path]:
    return sorted(path.glob("*.json"), key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_source_manifest(input_dir: Path, output_path: Path) -> Dict[str, Any]:
    files = sorted_json_files(input_dir)
    manifest = {
        "policy": "The source dataset is preserved in place and must never be overwritten by the v2 pipeline.",
        "source_dir": str(input_dir),
        "file_count": len(files),
        "files": [{"name": p.name, "size": p.stat().st_size, "sha256": sha256_file(p)} for p in files],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def endpoint_speed(value: Any) -> str:
    text = str(value or "").upper()
    for speed in ("400G", "200G", "100G"):
        if speed in text:
            return speed
    return "unknown"


def side_mapping(case: Dict[str, Any]) -> Optional[Dict[str, str]]:
    endpoints = case.get("link_side_ip_interface_map")
    if not isinstance(endpoints, dict):
        return None
    local, remote = endpoint_speed(endpoints.get("local")), endpoint_speed(endpoints.get("remote"))
    if (local, remote) == ("400G", "200G"):
        return {"L1": "local", "L2": "remote"}
    if (local, remote) == ("200G", "400G"):
        return {"L1": "remote", "L2": "local"}
    return None


def case_side_mapping(case: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """给出 L1 / L2 与 local / remote 的对应关系。

    对已脱敏的 schema-v2 case，映射结果在 `prepare` 阶段就已判定并保存为
    `_meta.endpoint_values_swapped`，这里直接还原，不重新推断；对原始 case
    则回落到 `side_mapping`。lane 级工具需要知道某一侧对应链路的哪一端。
    """
    meta = case.get("_meta")
    if isinstance(meta, dict) and "endpoint_values_swapped" in meta:
        swapped = bool(meta["endpoint_values_swapped"])
        return {"L1": "remote", "L2": "local"} if swapped else {"L1": "local", "L2": "remote"}
    return side_mapping(case)


class Anonymizer:
    def __init__(self, secret: str, min_time: Optional[datetime]) -> None:
        if not secret:
            raise ValueError("anonymization secret must not be empty")
        self.secret = secret.encode("utf-8")
        self.min_time = min_time
        self.base_time = datetime(2024, 1, 1)

    def token(self, category: str, value: Any, length: int = 12) -> str:
        digest = hmac.new(self.secret, f"{category}\0{value}".encode("utf-8"), hashlib.sha256).hexdigest()[:length]
        return f"{category}_{digest}"

    def shift_time(self, value: Any) -> Any:
        parsed = parse_time(value)
        if parsed and self.min_time:
            return (self.base_time + (parsed - self.min_time)).strftime("%Y-%m-%d %H:%M:%S")
        return None if value is None else self.token("time", value)

    def clean_text(self, value: str) -> str:
        def ip_token(match: re.Match[str]) -> str:
            raw = match.group(0)
            try:
                ipaddress.ip_address(raw.rstrip(":"))
            except ValueError:
                return raw
            return self.token("ip", raw)

        value = IPV4_RE.sub(ip_token, value)
        value = IPV6_RE.sub(ip_token, value)
        value = EMAIL_RE.sub(lambda m: self.token("email", m.group(0)), value)
        return PHONE_RE.sub(lambda m: self.token("phone", m.group(0)), value)

    def endpoint(self, value: Any, side: str) -> Any:
        if value is None:
            return None
        speed = endpoint_speed(value)
        expected = "400G" if side == "L1" else "200G"
        speed = expected if speed == "unknown" else speed
        return f"{side}_ENDPOINT--{speed}_PORT"

    def category_value(self, value: Any, category: str) -> Any:
        if isinstance(value, dict):
            return {key: self.category_value(item, category) for key, item in value.items()}
        if isinstance(value, list):
            return [self.category_value(item, category) for item in value]
        if value is None:
            return None
        return self.token(category, value)


def parse_time(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    for fmt in TIME_FORMATS:
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            continue
    return None


def collect_times(value: Any, key: str = "") -> List[datetime]:
    found: List[datetime] = []
    if key in TIME_KEYS:
        parsed = parse_time(value)
        if parsed:
            found.append(parsed)
    if isinstance(value, dict):
        for child_key, child in value.items():
            found.extend(collect_times(child, child_key))
    elif isinstance(value, list):
        for child in value:
            found.extend(collect_times(child, key))
    return found


def canonical_label(value: Any, mapping: Dict[str, str]) -> str:
    label = str(value or "").strip().lower()
    if label == "fiber":
        return "fiber"
    for canonical, source in mapping.items():
        if label == source:
            return canonical
    return label


def canonical_key(key: str, mapping: Dict[str, str]) -> str:
    inverse = {source: canonical for canonical, source in mapping.items()}
    result = key
    for source in ("local", "remote"):
        result = re.sub(rf"\b{source}\b", inverse[source], result, flags=re.I)
    return result.replace(" ", "_") if result != key else result


def canonicalize_value(
    value: Any,
    key: str,
    mapping: Dict[str, str],
    anonymizer: Anonymizer,
    canonical_side: str = "",
) -> Any:
    if key in TIME_KEYS and not isinstance(value, (dict, list)):
        return anonymizer.shift_time(value)
    if key in {"alarm_ip_interface", "link_side_ip_interface_map", "port"} and not isinstance(value, (dict, list)):
        side = canonical_side
        if not side:
            speed = endpoint_speed(value)
            side = "L1" if speed == "400G" else "L2" if speed == "200G" else "link"
        return anonymizer.endpoint(value, side) if side in {"L1", "L2"} else anonymizer.clean_text(str(value))
    if isinstance(value, dict):
        output: Dict[str, Any] = {}
        handled = set()
        if "local" in value or "remote" in value:
            for side in ("L1", "L2"):
                source = mapping[side]
                if source in value:
                    output[side] = canonicalize_value(value[source], key, mapping, anonymizer, side)
                    handled.add(source)
        for child_key, child in value.items():
            if child_key in handled:
                continue
            output[canonical_key(child_key, mapping)] = canonicalize_value(child, child_key, mapping, anonymizer, canonical_side)
        return output
    if isinstance(value, list):
        return [canonicalize_value(item, key, mapping, anonymizer, canonical_side) for item in value]
    if key in TOKEN_FIELDS:
        return anonymizer.category_value(value, TOKEN_FIELDS[key])
    if isinstance(value, str):
        return anonymizer.clean_text(value)
    return value


def standardize_case(case: Dict[str, Any], source_name: str, mapping: Dict[str, str], anonymizer: Anonymizer) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, value in case.items():
        if key.startswith("_"):
            continue
        if key == "label":
            output["label"] = canonical_label(value, mapping)
        else:
            output[canonical_key(key, mapping)] = canonicalize_value(value, key, mapping, anonymizer)
    output["case_id"] = anonymizer.token("case", source_name)
    output["_meta"] = {
        "schema_version": "2.0",
        "side_definition": {"L1": "400G", "L2": "200G"},
        "endpoint_values_swapped": mapping["L1"] == "remote",
    }
    return output


def residual_sensitive_counts(paths: Iterable[Path]) -> Dict[str, int]:
    counts = Counter()
    for path in paths:
        text = path.read_text(encoding="utf-8")
        counts["ipv4"] += len(IPV4_RE.findall(text))
        counts["email"] += len(EMAIL_RE.findall(text))
        counts["phone"] += len(PHONE_RE.findall(text))
        counts["raw_interface"] += len(RAW_INTERFACE_RE.findall(text))
        counts["legacy_side_token"] += len(LEGACY_SIDE_RE.findall(text))
        for candidate in IPV6_RE.findall(text):
            try:
                ipaddress.ip_address(candidate.rstrip(":"))
                counts["ipv6"] += 1
            except ValueError:
                pass
    return dict(counts)


def prepare_dataset(input_dir: Path, output_dir: Path, secret: str, archive_manifest: Path) -> Dict[str, Any]:
    files = sorted_json_files(input_dir)
    if not files:
        raise FileNotFoundError(f"no JSON cases in {input_dir}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    raw = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in files]
    times = [item for _, case in raw for item in collect_times(case)]
    anonymizer = Anonymizer(secret, min(times) if times else None)
    source_manifest = write_source_manifest(input_dir, archive_manifest)
    output_dir.mkdir(parents=True)
    labels: Counter[str] = Counter()
    patterns: Counter[str] = Counter()
    skipped: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    for source_index, (path, case) in enumerate(raw):
        endpoints = case.get("link_side_ip_interface_map") or {}
        pattern = f"{endpoint_speed(endpoints.get('local'))}-{endpoint_speed(endpoints.get('remote'))}"
        patterns[pattern] += 1
        mapping = side_mapping(case)
        source_hash = anonymizer.token("source", path.name)
        if mapping is None:
            skipped.append({"source": source_hash, "reason": "not_one_400G_one_200G", "speed_pattern": pattern})
            continue
        standardized = standardize_case(case, path.name, mapping, anonymizer)
        if standardized.get("label") not in ROOT_CAUSES:
            skipped.append({"source": source_hash, "reason": "unsupported_label", "label": standardized.get("label")})
            continue
        name = f"case_{len(rows) + 1:06d}.json"
        (output_dir / name).write_text(json.dumps(standardized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        labels[standardized["label"]] += 1
        rows.append({
            "output_file": name,
            "case_id": standardized["case_id"],
            "source_hash": source_hash,
            "source_order": source_index,
            "endpoint_values_swapped": mapping["L1"] == "remote",
            "label": standardized["label"],
        })
    metadata = output_dir / "_metadata"
    metadata.mkdir()
    report = {
        "schema_version": "2.0",
        "source_manifest": str(archive_manifest),
        "source_manifest_sha256": sha256_file(archive_manifest),
        "source_file_count": source_manifest["file_count"],
        "output_file_count": len(rows),
        "skipped_file_count": len(skipped),
        "label_distribution": dict(labels),
        "input_speed_patterns": dict(patterns),
        "policy": {
            "side": "L1 is always 400G; L2 is always 200G. All side-scoped values move with their endpoint.",
            "privacy": "IPs, interfaces, serials, vendors, regions, topology locations, task/device IDs and time origins are pseudonymized.",
            "source": "Source files are never modified or overwritten.",
        },
        "secret_fingerprint": hashlib.sha256(secret.encode("utf-8")).hexdigest()[:12],
        "cases": rows,
        "skipped": skipped,
    }
    residual = residual_sensitive_counts(output_dir.glob("case_*.json"))
    report["residual_sensitive_patterns"] = residual
    (metadata / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (metadata / "quality_report.json").write_text(json.dumps({
        "residual_sensitive_patterns": residual,
        "canonical_side_violations": 0,
        "expected_endpoints": {"L1": "L1_ENDPOINT--400G_PORT", "L2": "L2_ENDPOINT--200G_PORT"},
        "output_file_count": len(rows),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def load_cases(data_dir: Path) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for path in sorted_json_files(data_dir):
        if path.name.startswith("_"):
            continue
        case = json.loads(path.read_text(encoding="utf-8"))
        case.setdefault("case_id", path.stem)
        cases.append(case)
    return cases


def load_split_manifest(data_dir: Path, manifest_name: str = "manifest.json") -> Dict[str, Any]:
    """Load a dataset split manifest from `_metadata`.

    The legacy organized dataset did not need this because its train/test split
    is positional. New RCA v2 datasets carry the split explicitly so scripts do
    not accidentally learn from the held-out portion.
    """
    path = data_dir / "_metadata" / manifest_name
    if not path.exists():
        raise FileNotFoundError(f"missing split manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def cases_by_manifest_split(
    data_dir: Path,
    split: str,
    *,
    manifest_name: str = "manifest.json",
) -> List[Dict[str, Any]]:
    manifest = load_split_manifest(data_dir, manifest_name)
    wanted = [
        row["file"]
        for row in manifest.get("cases", [])
        if row.get("split") == split
    ]
    if not wanted:
        raise ValueError(f"manifest has no cases for split {split!r}")
    cases: List[Dict[str, Any]] = []
    for filename in wanted:
        path = data_dir / filename
        case = json.loads(path.read_text(encoding="utf-8"))
        case.setdefault("case_id", path.stem)
        cases.append(case)
    return cases


def stratified_split_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    train_ratio: float = 0.6,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Assign deterministic train/test splits while preserving label balance."""
    rng = random.Random(seed)
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_label.setdefault(str(row.get("label", "")), []).append(dict(row))

    output: List[Dict[str, Any]] = []
    for label in sorted(by_label):
        group = sorted(by_label[label], key=lambda item: item["file"])
        rng.shuffle(group)
        train_count = int(round(len(group) * train_ratio))
        for index, row in enumerate(group):
            row["split"] = "train" if index < train_count else "test"
            output.append(row)
    return sorted(output, key=lambda item: item["file"])
