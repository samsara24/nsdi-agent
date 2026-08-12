"""按 docs/EXPERT_EXPERIENCE.md 实现专家决策树，并在 organized_data 真实标签上验证。

两个变体：
  code — 文档第一部分（代码实现口径）的故障方向映射
  ai   — 文档第二部分整合前的 AI 提炼口径（多条规则方向相反）

用法：
  python scripts/validate_expert_rules.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

METRICS = ["rxpower", "txpower", "host_snr", "media_snr", "serdes_snr"]

# 文档 §3.3
THRESHOLDS: dict[str, dict[str, float]] = {
    "rxpower": {"down": -40, "low": -2.5, "high": 4.6, "diff": 1},
    "txpower": {"down": -40, "low": -2.5, "high": 2.5, "diff": 1.3},
    "host_snr": {"down": 0, "low": 22.8, "high": 27.5, "diff": 2.5},
    "media_snr": {"down": 0, "low": 22.4, "high": 28.7, "diff": 3},
    "serdes_snr": {"down": 0, "low": 458750, "high": 947750, "diff": 230000},
}

# 文档 §4，数值越小优先级越高
LEVEL = {"lane_down": 0, "low_value": 1, "high_value": 1, "lane_diff": 2}

# 文档 §5.3 单指标基础优先级
SINGLE_METRIC_BASE = {
    "host_snr": 2,
    "serdes_snr": 3,
    "media_snr": 4,
    "rxpower": 5,
    "txpower": 6,
}

# 故障方向：same = 异常所在端，opposite = 异常所在端的对端
DIRECTION_VARIANTS = {
    "code": {
        "txpower_lane_down": "same",
        "multi_metric": "opposite",
        "host_snr": "same",
        "serdes_snr": "same",
        "media_snr": "opposite",
        "rxpower": "opposite",
        "txpower": "same",
    },
    "ai": {
        "txpower_lane_down": "opposite",
        "multi_metric": "opposite",
        "host_snr": "opposite",
        "serdes_snr": "opposite",
        "media_snr": "opposite",
        "rxpower": "same",
        "txpower": "same",
    },
}

OPPOSITE = {"local": "remote", "remote": "local"}


def lane_values(case: dict[str, Any], metric: str, side: str) -> list[float]:
    block = case.get(metric)
    if not isinstance(block, dict):
        return []
    side_block = block.get(side)
    if not isinstance(side_block, dict):
        return []
    out = []
    for _, v in sorted(side_block.items(), key=lambda kv: str(kv[0])):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        out.append(float(v))
    return out


def side_metrics(case: dict[str, Any], side: str) -> dict[str, list[float]]:
    """取该端各指标的 lane 值，并施加 host_snr 特殊后处理（文档 §2.3）。"""
    values = {m: lane_values(case, m, side) for m in METRICS}
    if not any(v > 0 for v in values["host_snr"]):
        values["host_snr"] = []
    return values


def detect_anomaly(metric: str, values: list[float]) -> str | None:
    """文档 §3.1/§3.2：短路顺序 lane_down -> low_value -> high_value -> lane_diff。"""
    if not values:
        return None
    t = THRESHOLDS[metric]
    if any(v == t["down"] for v in values):
        return "lane_down"
    if any(v < t["low"] for v in values):
        return "low_value"
    if any(v > t["high"] for v in values):
        return "high_value"
    if max(values) - min(values) > t["diff"]:
        return "lane_diff"
    return None


def port_status(case: dict[str, Any], side: str) -> int:
    """文档 §1：txpower 与 rxpower 都异常时端口视为 down。"""
    abnormal = 0
    for metric in ("txpower", "rxpower"):
        values = lane_values(case, metric, side)
        if not values or not any(v > THRESHOLDS[metric]["down"] for v in values):
            abnormal += 1
    return 0 if abnormal == 2 else 1


def diagnose_side(
    case: dict[str, Any], side: str, direction: dict[str, str]
) -> dict[str, Any] | None:
    """单端诊断：检测异常 -> 匹配所有模式 -> 取 priority 最小者（文档 §5、§6.1）。"""
    anomalies = {}
    for metric in METRICS:
        atype = detect_anomaly(metric, side_metrics(case, side)[metric])
        if atype is not None:
            anomalies[metric] = atype

    def resolve(rule: str) -> str:
        return side if direction[rule] == "same" else OPPOSITE[side]

    candidates: list[dict[str, Any]] = []

    if anomalies.get("txpower") == "lane_down":
        candidates.append(
            {"priority": "0", "rule": "txpower_lane_down", "location": resolve("txpower_lane_down")}
        )

    if all(m in anomalies for m in ("serdes_snr", "media_snr", "rxpower")):
        candidates.append(
            {"priority": "1", "rule": "multi_metric", "location": resolve("multi_metric")}
        )

    for metric, base in SINGLE_METRIC_BASE.items():
        if metric in anomalies:
            candidates.append(
                {
                    "priority": f"{base}{LEVEL[anomalies[metric]]}",
                    "rule": f"single:{metric}",
                    "location": resolve(metric),
                }
            )

    if not candidates:
        return None
    # 文档 §5.3：priority 是字符串排序
    candidates.sort(key=lambda c: c["priority"])
    best = dict(candidates[0])
    best["side"] = side
    best["anomalies"] = anomalies
    return best


def diagnose(case: dict[str, Any], variant: str) -> dict[str, Any]:
    direction = DIRECTION_VARIANTS[variant]

    local_ps, remote_ps = port_status(case, "local"), port_status(case, "remote")
    if (local_ps, remote_ps) != (1, 1):
        gate = {(0, 1): "local", (1, 0): "remote", (0, 0): "local"}[(local_ps, remote_ps)]
        return {"prediction": gate, "priority": "gate", "source": "port_status_gate"}

    results = [
        r for r in (diagnose_side(case, "local", direction), diagnose_side(case, "remote", direction))
        if r is not None and r["location"] is not None
    ]
    results.sort(key=lambda r: r["priority"])

    if not results:
        return {"prediction": "local", "priority": "8", "source": "no_anomaly_fallback"}

    if (
        len(results) == 2
        and results[0]["location"] != results[1]["location"]
        and results[0]["priority"] == results[1]["priority"]
    ):
        return {"prediction": "fiber", "priority": "7", "source": "both_anomaly"}

    best = results[0]
    return {
        "prediction": best["location"],
        "priority": best["priority"],
        "source": f"{best['side']}:{best['rule']}",
    }


def load_cases(root: Path) -> list[dict[str, Any]]:
    cases = []
    for path in sorted(root.rglob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(case, dict) and case.get("label") in {"local", "remote", "fiber"}:
            case["_file"] = str(path.relative_to(root))
            cases.append(case)
    return cases


LABELS = ["local", "remote", "fiber"]


def evaluate(cases: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    matrix = {t: Counter() for t in LABELS}
    sources = Counter()
    source_correct = Counter()
    rows = []
    for case in cases:
        out = diagnose(case, variant)
        truth, pred = case["label"], out["prediction"]
        matrix[truth][pred] += 1
        sources[out["source"]] += 1
        if truth == pred:
            source_correct[out["source"]] += 1
        rows.append({"file": case["_file"], "label": truth, **out})

    correct = sum(matrix[t][t] for t in LABELS)
    total = len(cases)
    per_class = {}
    for t in LABELS:
        support = sum(matrix[t].values())
        pred_total = sum(matrix[o][t] for o in LABELS)
        tp = matrix[t][t]
        recall = tp / support if support else 0.0
        precision = tp / pred_total if pred_total else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[t] = {
            "support": support,
            "predicted": pred_total,
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
        }

    return {
        "variant": variant,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "confusion_matrix": {t: dict(matrix[t]) for t in LABELS},
        "per_class": per_class,
        "source_stats": {
            s: {"count": n, "correct": source_correct[s], "accuracy": round(source_correct[s] / n, 4)}
            for s, n in sources.most_common()
        },
        "rows": rows,
    }


def print_report(res: dict[str, Any]) -> None:
    print(f"\n{'=' * 68}\n变体 {res['variant']}  准确率 {res['correct']}/{res['total']} = {res['accuracy']:.2%}\n{'=' * 68}")
    print(f"{'真值\\预测':<12}{'local':>8}{'remote':>8}{'fiber':>8}{'recall':>9}")
    for t in LABELS:
        row = res["confusion_matrix"][t]
        print(
            f"{t:<12}{row.get('local', 0):>8}{row.get('remote', 0):>8}"
            f"{row.get('fiber', 0):>8}{res['per_class'][t]['recall']:>9.2%}"
        )
    print("\n按裁决来源:")
    for src, st in res["source_stats"].items():
        print(f"  {src:<32}{st['count']:>4} 例  命中 {st['correct']:>3}  {st['accuracy']:.2%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="organized_data")
    ap.add_argument("--out", default="artifacts/expert_rule_validation")
    args = ap.parse_args()

    root = Path(args.data_dir)
    cases = load_cases(root)
    dist = Counter(c["label"] for c in cases)
    print(f"载入 {len(cases)} 例，标签分布 {dict(dist)}")
    majority, majority_n = dist.most_common(1)[0]
    print(f"多数类基线（全判 {majority}）: {majority_n}/{len(cases)} = {majority_n / len(cases):.2%}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for variant in ("code", "ai"):
        res = evaluate(cases, variant)
        print_report(res)
        (out_dir / f"predictions_{variant}.json").write_text(
            json.dumps(res.pop("rows"), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary[variant] = res

    summary["dataset"] = {"total": len(cases), "distribution": dict(dist),
                          "majority_baseline": round(majority_n / len(cases), 4)}
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n结果写入 {out_dir}")


if __name__ == "__main__":
    main()
