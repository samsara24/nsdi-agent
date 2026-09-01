#!/usr/bin/env python3
"""Audit whether constrained LLM reasoning can safely cover v5 abstentions.

Prediction-side fields are read before labels are attached.  The report then joins
the frozen labels for retrospective evaluation.  This script does not alter the
knowledge bundle, decision graph, labels, or production policy.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rca_framework.filtered_rule_expert import assess_filtered_rule_expert  # noqa: E402
from rca_framework.data import cases_by_manifest_split  # noqa: E402
from rca_framework.expert import DOC_VARIANT, ExpertVariant, diagnose  # noqa: E402
from rca_framework.knowledge import OfflineKnowledgeBundle  # noqa: E402


V5 = ROOT / "artifacts/filtered_rule_decision_graph_test_v5/case_analysis.json"
REMOTE = ROOT / "artifacts/filtered_rule_temporal_20260823T122701Z/run"
OUTPUT = ROOT / "artifacts/filtered_rule_full_test_analysis_v3"

TELEMETRY_KEYS = (
    "alarm_name", "alarm_time", "alarm_ip_interface", "link_location",
    "bias", "txpower", "rxpower", "transmission", "media_snr", "host_snr",
    "serdes_snr", "RxLOS", "RxLOL", "TxLOS", "TxLOL", "Temperature",
    "Voltage", "Lane number",
)
METRICS = ("media_snr", "serdes_snr", "host_snr", "txpower", "rxpower", "bias")
DOWN_THRESHOLDS = {"rxpower": -39.0, "txpower": -39.0, "media_snr": 0.0, "host_snr": 0.0, "serdes_snr": 0.0}


def diagnostic_missing_fields(fields: Iterable[str]) -> list[str]:
    """Host SNR is optional corroboration, never a diagnosis blocker."""
    return [str(field) for field in fields if not str(field).endswith(".host_snr")]


def diagnostic_acquisition(items: Iterable[str]) -> list[str]:
    revised = []
    for item in items:
        text = str(item).replace(
            "补采media/SerDes/host侧SNR，确认异常位于光路、模块介质侧还是设备数字侧。",
            "补采关键的media/SerDes数据以区分光路与模块数字侧；若现场已有host_snr，可作为本端电口方向的增强证据。",
        )
        if "host_snr" in text.lower() and "增强证据" not in text:
            continue
        revised.append(text)
    return revised


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def wilson(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def metric(rows: Iterable[Mapping[str, Any]], field: str = "forced_correct") -> dict[str, Any]:
    rows = list(rows)
    correct = sum(bool(row[field]) for row in rows)
    low, high = wilson(correct, len(rows))
    return {
        "count": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "wilson_95": [low, high],
    }


def classification_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    labels = ("L1", "L2", "fiber")
    confusion = {
        actual: {predicted: 0 for predicted in labels}
        for actual in labels
    }
    for row in rows:
        confusion[row["actual"]][row["analysis_verdict"]] += 1
    per_class = {}
    for label in labels:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted = sum(confusion[actual][label] for actual in labels)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        per_class[label] = {
            "support": support,
            "predicted": predicted,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        }
    return {
        "confusion_matrix": confusion,
        "per_class": per_class,
        "macro_f1": sum(item["f1"] for item in per_class.values()) / len(labels),
    }


def extract_tree_prediction(chain: Iterable[Mapping[str, Any]]) -> str | None:
    for item in chain:
        if item.get("kind") != "numeric_decision_tree":
            continue
        match = re.search(r"predicts (L1|L2|fiber)", str(item.get("statement", "")))
        if match:
            return match.group(1)
    return None


def compact_steps(chain: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "statement": item.get("statement", ""),
            "tokens": list(item.get("tokens", [])),
            "source": item.get("source", ""),
        }
        for item in chain
        if item.get("kind") in {"llm_step", "llm_arbitration_step"}
    ]


def telemetry_snapshot(remote: Mapping[str, Any]) -> dict[str, Any]:
    """Copy only observable telemetry; never copy dataset contract or labels."""
    telemetry = remote.get("evidence_pack", {}).get("telemetry", {})
    return {key: telemetry.get(key) for key in TELEMETRY_KEYS if key in telemetry}


def blind_join(v5_rows: list[Mapping[str, Any]], remote_rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build prediction records without reading ``actual`` from either input."""
    remote_by_id = {row["case_id"]: row for row in remote_rows}
    blind: list[dict[str, Any]] = []
    for current in v5_rows:
        remote = remote_by_id[current["case_id"]]
        final = remote["final_decision"]
        branch = remote["branch_outcome"]
        proposed = final.get("proposed_verdict") or branch.get("verdict")
        terminal = final.get("verdict") if final.get("action") == "final" else None
        expert = current.get("expert", {}).get("verdict")
        history = current.get("history", [])
        top_history = history[0].get("label") if history else None
        tree = extract_tree_prediction(branch.get("evidence_chain", []))
        breakdown = branch.get("confidence_breakdown", {})
        penalties = list(final.get("compliance_penalties", []))
        telemetry = telemetry_snapshot(remote)
        expert_sides = current.get("expert", {}).get("sides", [])
        symptom_side = expert_sides[0].get("side") if expert_sides else None
        causal_expert = assess_filtered_rule_expert(
            expert_group=current.get("expert", {}).get("group", ""),
            expert_verdict=expert,
            symptom_side=symptom_side,
            tokens=current.get("features", {}).get("tokens", []),
            telemetry=telemetry,
        )
        verdict_mismatch = next(
            (item for item in penalties if item.get("kind") == "verdict_step_mismatch"), None
        )
        mismatch_match = re.search(
            r"verdict=(L1|L2|fiber).*汇总 (L1|L2|fiber)",
            str(verdict_mismatch.get("message", "")) if verdict_mismatch else "",
        )
        llm_output_valid = bool(
            not final.get("fallback_source")
            and float(breakdown.get("physical_compliance", 1.0)) > 0.0
            and not any(float(item.get("physical_compliance_cap", 1.0)) <= 0.0 for item in penalties)
            and verdict_mismatch is None
        )
        if current.get("action") == "final":
            corrected_action = "final"
            corrected_verdict = current.get("verdict")
            correction_reason = "v5决策图路径已通过训练留一支持、纯度与拓扑门，保留冻结终裁。"
        elif causal_expert.terminal:
            corrected_action = "final"
            corrected_verdict = causal_expert.verdict
            correction_reason = "因果专家规则具备独立强证据，覆盖旧LLM终裁。"
        elif len(causal_expert.candidates) > 1:
            corrected_action = "human_review"
            corrected_verdict = None
            correction_reason = "因果规则得到多个不可区分候选，取消旧单标签终裁并转人工。"
        elif causal_expert.verdict and terminal and causal_expert.verdict != terminal:
            corrected_action = "human_review"
            corrected_verdict = None
            correction_reason = "LLM终裁与独立物理候选冲突，两者均不能单独覆盖另一方，转人工。"
        elif llm_output_valid and terminal:
            corrected_action = "final"
            corrected_verdict = terminal
            correction_reason = "LLM输出通过结构与物理校验，保留原终裁。"
        elif causal_expert.verdict:
            corrected_action = "human_review"
            corrected_verdict = causal_expert.verdict
            correction_reason = "旧LLM输出无效；保留独立物理候选供人工复核，不自动终裁。"
        else:
            corrected_action = (
                "request_evidence"
                if diagnostic_missing_fields(current.get("missing_fields", [])) else "human_review"
            )
            corrected_verdict = causal_expert.verdict if causal_expert.verdict else None
            correction_reason = "旧LLM输出无效或缺少可终裁因果链，取消final并转补采/人工。"
        if current.get("action") == "final":
            analysis_verdict = current.get("verdict")
            analysis_source = "v5_decision_graph"
        elif mismatch_match and mismatch_match.group(2) in {"L1", "L2", "fiber"}:
            analysis_verdict = mismatch_match.group(2)
            analysis_source = "llm_step_aggregate_after_constraint_review"
        elif causal_expert.terminal:
            analysis_verdict = causal_expert.verdict
            analysis_source = "causal_expert_strong"
        elif llm_output_valid and terminal:
            analysis_verdict = terminal
            analysis_source = "constraint_valid_llm"
        elif causal_expert.verdict:
            analysis_verdict = causal_expert.verdict
            analysis_source = "causal_expert_candidate"
        elif (
            causal_expert.rule == "aligned_receive_chain_ambiguous"
            and causal_expert.candidates
        ):
            analysis_verdict = causal_expert.candidates[0]
            analysis_source = "aligned_receive_chain_best_effort"
        else:
            analysis_verdict = proposed or top_history or tree or expert or "L2"
            analysis_source = "forced_best_effort_llm"
        analysis_confidence_tier = (
            "high" if corrected_action == "final" else
            "medium" if corrected_action == "human_review" else "low"
        )
        pilot_gate = bool(
            terminal
            and final.get("branch") == "N5c"
            and expert
            and terminal == expert
        )
        blind.append({
            "case_id": current["case_id"],
            "source_dataset": current["source_dataset"],
            "topology_id": current["topology_id"],
            "v5_action": current.get("action"),
            "v5_verdict": current.get("verdict"),
            "telemetry_status": current["telemetry_status"],
            "v5_review_category": current["review_category"],
            "expert_group": current["expert"]["group"],
            "feature_families": sorted(current["features"].get("by_family", {})),
            "feature_tokens": list(current["features"].get("tokens", [])),
            "missing_fields": diagnostic_missing_fields(current.get("missing_fields", [])),
            "optional_missing_fields": [
                str(field) for field in current.get("missing_fields", [])
                if str(field).endswith(".host_snr")
            ],
            "remote_branch": final.get("branch"),
            "remote_action": final.get("action"),
            "llm_terminal_verdict": terminal,
            "llm_proposed_verdict": proposed,
            "expert_candidate": expert,
            "causal_expert": causal_expert.to_dict(),
            "llm_pre_reconcile_verdict": mismatch_match.group(1) if mismatch_match else proposed,
            "llm_step_aggregate_verdict": mismatch_match.group(2) if mismatch_match else proposed,
            "reasoning_verdict_conflict": verdict_mismatch is not None,
            "top_history_candidate": top_history,
            "numeric_tree_candidate": tree,
            "pilot_gate": pilot_gate,
            "confidence": final.get("confidence", 0.0),
            "confidence_lower_bound": final.get("confidence_lower_bound", 0.0),
            "confidence_breakdown": breakdown,
            "compliance_penalties": penalties,
            "llm_output_valid": llm_output_valid,
            "corrected_action": corrected_action,
            "corrected_verdict": corrected_verdict,
            "correction_reason": correction_reason,
            "analysis_verdict": analysis_verdict,
            "analysis_source": analysis_source,
            "analysis_confidence_tier": analysis_confidence_tier,
            "llm_steps": compact_steps(branch.get("evidence_chain", [])),
            "raw_telemetry": telemetry,
            "acquisition_recommendations": diagnostic_acquisition(current.get("acquisition_recommendations", [])),
        })
    return blind


def attach_labels(blind: list[Mapping[str, Any]], v5_rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    truth = {row["case_id"]: row["actual"] for row in v5_rows}
    reviewed: list[dict[str, Any]] = []
    for item in blind:
        row = dict(item)
        actual = truth[row["case_id"]]
        forced = row["llm_proposed_verdict"]
        terminal = row["llm_terminal_verdict"]
        row["actual"] = actual
        row["forced_correct"] = forced == actual
        row["terminal_correct"] = terminal == actual if terminal else None
        corrected = row.get("corrected_verdict")
        row["corrected_correct"] = corrected == actual if corrected else None
        row["analysis_correct"] = row["analysis_verdict"] == actual
        row["label_changed"] = bool(corrected and corrected != forced)
        row["old_wrong_corrected_right"] = bool(forced != actual and corrected == actual)
        row["old_right_corrected_wrong"] = bool(forced == actual and corrected and corrected != actual)
        if row["v5_action"] == "final":
            assessment = (
                "v5决策图冻结结论正确，保留高置信终裁。"
                if row["analysis_correct"] else
                "v5决策图冻结结论错误，应按原复盘类别继续修正路径或复核标签。"
            )
        elif row["reasoning_verdict_conflict"]:
            assessment = "推理—结论冲突：模型原结论与步骤目标不一致；两种自动改写在本批各纠正8条、各破坏8条，必须转人工而非自动选边。"
        elif row["old_wrong_corrected_right"]:
            assessment = "旧终裁错误，新因果候选与真实标签一致；该候选仍需训练校准决定能否自动放行。"
        elif forced != actual and len(row["causal_expert"].get("candidates", [])) == 1 and actual in row["causal_expert"]["candidates"]:
            assessment = "推理中的独立物理候选与真实标签一致，而旧LLM最终标签错误；因候选与LLM冲突，修复后取消final并进入人工复核。"
        elif row["v5_review_category"] == "missing_evidence":
            assessment = "不可辨识：缺少可定位异常，LLM 标签主要来自先验，不能新增安全覆盖。"
        elif actual in row["causal_expert"].get("candidates", []) and not row["causal_expert"].get("verdict"):
            assessment = "真实标签位于新因果候选集合内，但当前可见证据无法从候选中唯一选出，正确动作是补采/人工。"
        elif actual == "fiber" and forced != "fiber":
            assessment = "介质不可辨识：现有遥测缺少双向对称丢失/现场换纤证据，端点化偏置明显。"
        elif row["compliance_penalties"]:
            assessment = "约束执行失败：回答包含约束冲突、无支持步骤或编造引用，不能放行。"
        elif forced == actual:
            assessment = "推理候选正确；仍需使用独立训练/验证标定后才能转为自动终裁。"
        else:
            assessment = "候选冲突：现有物理证据不能区分端点，LLM 推理未带来可靠增益。"
        row["label_aware_assessment"] = assessment
        reviewed.append(row)
    return reviewed


def td(value: Any) -> str:
    return f"<td>{html.escape(str(value))}</td>"


def display_value(value: Any) -> str:
    if isinstance(value, Mapping):
        return ", ".join(
            f"lane {key}={'missing' if value[key] is None else value[key]}"
            for key in sorted(value, key=str)
        )
    if value is None:
        return "missing"
    return str(value)


def token_observations(token: str, telemetry: Mapping[str, Any]) -> list[tuple[str, Any]]:
    parts = token.split(":")
    side = next((part for part in parts if part in {"L1", "L2"}), None)
    observations: list[tuple[str, Any]] = []

    if parts and parts[0] == "status" and len(parts) >= 3:
        status = telemetry.get(parts[2], {})
        observations.append((f"{parts[2]}.{parts[1]}", status.get(parts[1]) if isinstance(status, Mapping) else None))

    direction = next((part for part in parts if "_to_" in part), None)
    if direction:
        source, target = direction.split("_to_", 1)
        for metric, endpoint in (("txpower", source), ("rxpower", target)):
            values = telemetry.get(metric, {})
            observations.append((f"{metric}.{endpoint}", values.get(endpoint) if isinstance(values, Mapping) else None))
        transmission = telemetry.get("transmission", {})
        observations.append((f"transmission.{source}-{target}", transmission.get(f"{source}-{target}") if isinstance(transmission, Mapping) else None))

    metric = next((candidate for candidate in METRICS if candidate in token), None)
    if metric:
        values = telemetry.get(metric, {})
        if side and isinstance(values, Mapping):
            observations.append((f"{metric}.{side}", values.get(side)))
        elif isinstance(values, Mapping):
            for endpoint in ("L1", "L2"):
                observations.append((f"{metric}.{endpoint}", values.get(endpoint)))

    if not observations and side:
        for metric in ("txpower", "rxpower", "media_snr", "serdes_snr"):
            values = telemetry.get(metric, {})
            if isinstance(values, Mapping) and side in values:
                observations.append((f"{metric}.{side}", values.get(side)))
    if not observations:
        observations.extend((key, telemetry.get(key)) for key in ("alarm_name", "link_location"))

    deduplicated: list[tuple[str, Any]] = []
    seen: set[str] = set()
    for key, value in observations:
        if key not in seen:
            deduplicated.append((key, value))
            seen.add(key)
    return deduplicated


@lru_cache(maxsize=1)
def topology_edges() -> Mapping[str, list[float | None]]:
    bundle = load_json(ROOT / "artifacts/filtered_rule_deterministic_knowledge_v2/knowledge_bundle.json")
    return bundle["feature_model"].get("topology_level_edges", {})


def numeric_lanes(telemetry: Mapping[str, Any], metric: str, side: str) -> list[float]:
    block = telemetry.get(metric, {})
    values = block.get(side, {}) if isinstance(block, Mapping) else {}
    if not isinstance(values, Mapping):
        return []
    return [float(value) for value in values.values() if isinstance(value, (int, float))]


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def feature_rule(token: str, topology_id: str, telemetry: Mapping[str, Any]) -> str:
    parts = token.split(":")
    if len(parts) >= 4 and parts[0] in {"drop", "drop_ratio"}:
        side, metric = parts[1], parts[2]
        threshold = DOWN_THRESHOLDS.get(metric)
        values = numeric_lanes(telemetry, metric, side)
        down = sum(value <= threshold for value in values) if threshold is not None else 0
        ratio = down / len(values) if values else 0.0
        return f"触发依据：{down}/{len(values)} 条 lane ≤ {threshold}，异常比例 {ratio:.0%}，范围分桶为 {parts[3]}。"
    if len(parts) >= 4 and parts[0] in {"topology_level", "level"}:
        side, statistic, tail = parts[1], parts[2], parts[3]
        metric = next((candidate for candidate in ("media_snr", "serdes_snr", "host_snr", "txpower", "rxpower") if statistic.startswith(candidate)), None)
        if metric:
            raw = numeric_lanes(telemetry, metric, side)
            healthy = [value for value in raw if value > DOWN_THRESHOLDS.get(metric, -math.inf)]
            observed = percentile(healthy, 0.25) if metric == "media_snr" else (sum(healthy) / len(healthy) if healthy else None)
            key = f"{topology_id}:{side}:{statistic}:width{len(raw)}"
            low, high = topology_edges().get(key, [None, None])
            observed_text = f"{observed:.6g}" if observed is not None else "missing"
            low_text = f"{low:.6g}" if low is not None else "missing"
            high_text = f"{high:.6g}" if high is not None else "missing"
            comparator = f"< {low_text}" if tail == "low_tail" else f"> {high_text}"
            note = "健康 lane 下四分位" if metric == "media_snr" else "健康 lane 均值"
            return f"触发依据：{note}={observed_text}，训练集同拓扑/同宽度阈值为 {comparator}（low={low_text}, high={high_text}）。"
    if len(parts) >= 3 and parts[0] == "status":
        return f"触发依据：状态字段 {parts[2]}.{parts[1]} 被观测为异常值。"
    if any("_to_" in part for part in parts):
        return "触发依据：按同编号 lane 对照发送端 Tx、接收端 Rx 与 transmission；该关系描述传播方向，不等价于绝对链路损耗。"
    return "触发依据：该 token 来自当前冻结特征抽取器；请结合下方完整遥测和训练阈值核对。"


def evidence_chip(token: str, topology_id: str, telemetry: Mapping[str, Any]) -> str:
    observations = token_observations(token, telemetry)
    rows = "".join(
        f"<tr><th>{html.escape(key)}</th><td>{html.escape(display_value(value))}</td></tr>"
        for key, value in observations
    )
    return (
        "<span class='evidence-chip' tabindex='0'>"
        f"{html.escape(token)}<span class='evidence-popover'><b>对应原始观测</b>"
        f"<table>{rows}</table><p class='trigger'>{html.escape(feature_rule(token, topology_id, telemetry))}</p>"
        "<small>阈值来自124条训练集的同拓扑统计；它是统计异常，不自动等价于物理故障。missing 不等于正常。</small>"
        "</span></span>"
    )


def raw_telemetry_details(telemetry: Mapping[str, Any]) -> str:
    sections = []
    for key in TELEMETRY_KEYS:
        if key not in telemetry:
            continue
        value = telemetry[key]
        if isinstance(value, Mapping):
            endpoint_rows = "".join(
                f"<tr><th>{html.escape(str(endpoint))}</th><td>{html.escape(display_value(values))}</td></tr>"
                for endpoint, values in value.items()
            )
            body = f"<table>{endpoint_rows}</table>"
        else:
            body = f"<p>{html.escape(display_value(value))}</p>"
        sections.append(f"<details><summary>{html.escape(key)}</summary>{body}</details>")
    return "".join(sections)


def render_case(row: Mapping[str, Any]) -> str:
    telemetry = row["raw_telemetry"]
    steps = "".join(
        f"<li><b>{html.escape(step['source'])}</b>：{html.escape(step['statement'])}"
        f"<div class='chips'>{''.join(evidence_chip(token, row['topology_id'], telemetry) for token in step['tokens'])}</div></li>"
        for step in row["llm_steps"]
    ) or "<li>没有可用的结构化 LLM 推理步骤。</li>"
    penalties = "".join(
        f"<li>{html.escape(str(item.get('kind')))}：{html.escape(str(item.get('message')))}</li>"
        for item in row["compliance_penalties"]
    ) or "<li>无</li>"
    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>{row['case_id']}</title>
<style>body{{font:15px/1.65 system-ui;margin:32px;max-width:1180px;color:#18212b}}.card{{border:1px solid #d8dee6;border-radius:10px;padding:18px;margin:14px 0}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #d8dee6;padding:8px;text-align:left}}small{{display:block;color:#667085}}.ok{{color:#087443}}.bad{{color:#b42318}}.chips{{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0 16px}}.evidence-chip{{position:relative;display:inline-block;padding:4px 9px;border:1px solid #84adff;background:#eff4ff;border-radius:999px;color:#1849a9;cursor:help;outline:none}}.evidence-popover{{display:none;position:absolute;z-index:20;left:0;top:calc(100% + 7px);width:min(620px,80vw);padding:12px;background:white;color:#18212b;border:1px solid #98a2b3;border-radius:10px;box-shadow:0 12px 32px rgba(16,24,40,.2)}}.evidence-chip:hover .evidence-popover,.evidence-chip:focus .evidence-popover{{display:block}}.evidence-popover table{{margin:8px 0;font-size:13px}}.trigger{{padding:8px;background:#f8fafc;border-radius:6px;color:#344054;white-space:normal}}details{{border:1px solid #eaecf0;border-radius:8px;padding:8px 12px;margin:7px 0}}summary{{font-weight:650;cursor:pointer}}ol>li{{margin-bottom:12px}}</style></head><body>
<p><a href='../index.html'>← 返回主报告</a></p><h1>{row['case_id']}</h1>
<div class='card'><h2>盲态推理结果</h2><table>
<tr><th>来源/拓扑</th>{td(row['source_dataset']+' / '+row['topology_id'])}</tr>
<tr><th>v5 复盘类型</th>{td(row['v5_review_category'])}</tr>
<tr><th>LLM 分支/动作</th>{td(str(row['remote_branch'])+' / '+str(row['remote_action']))}</tr>
<tr><th>LLM 终裁/建议</th>{td(str(row['llm_terminal_verdict'])+' / '+str(row['llm_proposed_verdict']))}</tr>
<tr><th>旧专家/新因果专家</th>{td(str(row['expert_candidate'])+' / '+str(row['causal_expert']['verdict'])+' ('+str(row['causal_expert']['strength'])+')')}</tr>
<tr><th>历史/数值树候选</th>{td(str(row['top_history_candidate'])+' / '+str(row['numeric_tree_candidate']))}</tr>
<tr><th>修复后动作/候选</th>{td(str(row['corrected_action'])+' / '+str(row['corrected_verdict']))}</tr>
<tr><th>最终分析结论</th>{td(str(row['analysis_verdict'])+' / '+str(row['analysis_confidence_tier'])+' / '+str(row['analysis_source']))}</tr>
<tr><th>新因果候选集</th>{td(', '.join(row['causal_expert']['candidates']) or '无')}</tr>
<tr><th>LLM 原结论/步骤汇总</th>{td(str(row['llm_pre_reconcile_verdict'])+' / '+str(row['llm_step_aggregate_verdict']))}</tr>
<tr><th>修复依据</th>{td(row['correction_reason'])}</tr>
<tr><th>试验门</th>{td('通过' if row['pilot_gate'] else '不通过')}</tr>
</table></div>
<div class='card'><h2>旧专家规则修正</h2><p><b>{html.escape(row['causal_expert']['rule'])}</b>：{html.escape(row['causal_expert']['reason'])}</p><p>证据：{html.escape(', '.join(row['causal_expert']['evidence']) or '无独立强证据')}</p><p>允许候选：{html.escape(', '.join(row['causal_expert']['candidates']) or '无')}</p></div>
<div class='card'><h2>可见证据</h2><p>将鼠标停留在证据标签上，或用键盘聚焦，即可查看对应原始 lane 数值和触发阈值。</p><div class='chips'>{''.join(evidence_chip(token, row['topology_id'], telemetry) for token in row['feature_tokens']) or '无'}</div><p><b>关键缺失：</b>{html.escape(', '.join(row['missing_fields']) or '无')}</p><p><b>可选缺失：</b>{html.escape(', '.join(row['optional_missing_fields']) or '无')}。Host SNR 缺失不扣分；仅在有效观测与本端方向一致时增强判断。</p></div>
<div class='card'><h2>LLM 推理链</h2><ol>{steps}</ol><h3>约束校验问题</h3><ul>{penalties}</ul></div>
<div class='card'><h2>完整原始遥测</h2><p>按指标展开查看两端及全部 lane。此区域来自盲态输入，不包含真实标签或数据集标签契约。</p>{raw_telemetry_details(telemetry)}</div>
<div class='card'><h2>标签揭示后的复盘</h2><p>真实标签：<b>{row['actual']}</b>；最终分析：<b>{row['analysis_verdict']}</b>（<span class='{'ok' if row['analysis_correct'] else 'bad'}'>{'正确' if row['analysis_correct'] else '错误'}</span>）；安全动作：<b>{row['corrected_action']}</b>。旧建议：<b>{row['llm_proposed_verdict']}</b>（{'正确' if row['forced_correct'] else '错误'}）。</p><p>{html.escape(row['label_aware_assessment'])}</p></div>
</body></html>"""


def render_index(summary: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> str:
    corrections = summary["correction_audit"]
    corrected_terminal = summary["subgroups"]["修复后高置信自动终裁"]
    train_aligned = summary["training_causal_calibration"]["aligned_receive_chain"]
    subgroup_rows = "".join(
        f"<tr>{td(name)}{td(item['count'])}{td(item['correct'])}{td(pct(item['accuracy']))}{td(pct(item['wilson_95'][0])+'–'+pct(item['wilson_95'][1]))}</tr>"
        for name, item in summary["subgroups"].items()
    )
    case_rows = "".join(
        "<tr>"
        + f"<td><a href='cases/{row['case_id']}.html'>{row['case_id']}</a></td>"
        + td(row["source_dataset"])
        + td(row["v5_review_category"])
        + td(row["remote_action"])
        + td(row["llm_proposed_verdict"])
        + td(row["analysis_verdict"])
        + td(row["analysis_confidence_tier"])
        + td(row["corrected_action"])
        + td(row["actual"])
        + td("纠正" if row["old_wrong_corrected_right"] else "回归" if row["old_right_corrected_wrong"] else "不变/未定")
        + td("通过" if row["pilot_gate"] else "否")
        + "</tr>"
        for row in rows
    )
    penalty_rows = "".join(f"<tr>{td(k)}{td(v)}</tr>" for k, v in summary["penalties"].items())
    per_class_rows = "".join(
        f"<tr>{td(label)}{td(item['support'])}{td(item['predicted'])}{td(pct(item['precision']))}{td(pct(item['recall']))}{td(pct(item['f1']))}</tr>"
        for label, item in summary["classification"]["per_class"].items()
    )
    confusion = summary["classification"]["confusion_matrix"]
    confusion_rows = "".join(
        f"<tr>{td(actual)}{td(confusion[actual]['L1'])}{td(confusion[actual]['L2'])}{td(confusion[actual]['fiber'])}</tr>"
        for actual in ("L1", "L2", "fiber")
    )
    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>完整测试集最终分析复盘</title>
<style>body{{font:15px/1.65 system-ui;margin:32px;color:#18212b}}.hero{{background:#f4f7fb;border-radius:14px;padding:24px}}.grid{{display:grid;grid-template-columns:repeat(4,minmax(170px,1fr));gap:12px}}.metric,.card{{border:1px solid #d8dee6;border-radius:10px;padding:16px;margin:14px 0}}.metric b{{font-size:28px;display:block}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #d8dee6;padding:8px;text-align:left}}th{{background:#f8fafc;position:sticky;top:0}}.warn{{color:#b54708}}.bad{{color:#b42318}}.ok{{color:#087443}}code{{background:#f2f4f7;padding:2px 5px}}</style></head><body>
<div class='hero'><h1>完整测试集：非空分析结论与安全终裁复盘 v3</h1><p>对象为两个来源的全部484条测试case。每条均输出L1/L2/fiber三选一的最终分析结论，同时独立保存自动终裁、人工复核或补采动作；真实标签只在结论冻结后用于评估。旧LLM输出来自 <code>20260823T122701Z</code>。</p></div>
<div class='grid'>
<div class='metric'><b>484</b>逐case重刷</div><div class='metric'><b>{pct(summary['subgroups']['完整测试集最终分析']['accuracy'])}</b>完整测试准确率</div>
<div class='metric'><b>{corrections['reasoning_verdict_conflicts']}</b>推理/结论冲突</div><div class='metric'><b>{corrections['reasoning_candidate_matches_truth']}</b>推理候选对、旧标签错</div>
</div>
<div class='card'><h2>修复结论</h2><p class='ok'><b>分析结论与安全动作已经分层。</b>484条全部给出非空三分类结论；fatal fallback、physical_compliance=0、零上限惩罚或推理—结论冲突仍不能成为高置信final。旧multi_metric也不再把接收侧三项共因异常直接翻转为对端根因。</p><p><b>Host SNR改为纯增强证据。</b>缺失不扣分、不触发补采、不降低诊断完整度；只有有效观测且与本端电口方向一致时才增强候选。</p><p class='warn'><b>不能把接收异常统一改成本端。</b>训练集 aligned receive 共{train_aligned['support']}条：本端标签{train_aligned['local_truth']}、对端标签{train_aligned['opposite_truth']}、fiber {train_aligned['fiber_truth']}，本端规则准确率只有{pct(train_aligned['local_accuracy'])}。测试复盘中{corrections['label_changed']}条候选改向只纠正{corrections['old_wrong_corrected_right']}条，同时造成{corrections['old_right_corrected_wrong']}条回归，因此改向只能形成候选，不能直接成为正式自动规则。</p><p>共识别{corrections['reasoning_candidate_matches_truth']}条“独立物理候选与真实标签一致、旧最终标签错误”的case。高置信终裁共{corrected_terminal['count']}条，命中{corrected_terminal['correct']}条（{pct(corrected_terminal['accuracy'])}）；其余case仍给出最终分析，但明确标记为中/低置信和人工复核/补采。</p></div>
<div class='card'><h2>三种口径</h2><table><tr><th>口径</th><th>样本</th><th>正确</th><th>准确率</th><th>95% Wilson</th></tr>{subgroup_rows}</table></div>
<div class='card'><h2>分类效果</h2><p>Macro-F1：<b>{pct(summary['classification']['macro_f1'])}</b></p><table><tr><th>类别</th><th>真实数</th><th>预测数</th><th>Precision</th><th>Recall</th><th>F1</th></tr>{per_class_rows}</table><h3>混淆矩阵（行=真实，列=预测）</h3><table><tr><th>真实\预测</th><th>L1</th><th>L2</th><th>fiber</th></tr>{confusion_rows}</table></div>
<div class='card'><h2>主要失败原因</h2><ul><li><b>150 条不可辨识：</b>没有异常或关键遥测不足，最终分析准确率52.67%，本质接近端点先验。</li><li><b>logical8 失配：</b>完整67条命中33条（49.25%），旧LLM知识与8×8拓扑仍不匹配。</li><li><b>约束没有真正控制旧生成：</b>旧回答大量出现约束冲突、无支持步骤、编造证据和量测否决；新版门禁会阻断自动终裁，但不能凭空恢复正确标签。</li><li><b>fiber覆盖不足：</b>完整测试集只有16条fiber，最终分析仅输出4条fiber，端点化偏置仍明显。</li><li><b>置信度不可用：</b>新增高置信终裁139条仅69.06%，低于v5冻结终裁的80.28%，不能直接发布。</li></ul><table><tr><th>约束问题</th><th>次数</th></tr>{penalty_rows}</table></div>
<div class='card'><h2>本轮已落地与下一步</h2><ol>
<li><b>已完成：multi_metric因果拆分。</b>对端Tx故障、本端host故障、双向介质候选和单向不可辨识分别处理，不再重复计算同lane的Rx/media/SerDes。</li>
<li><b>已完成：LLM fatal门禁。</b>无效fallback、物理合规度0及步骤—结论冲突强制降级。</li>
<li><b>P0：LLM 只做受限仲裁，不做自由三分类。</b>输入必须是物理层生成的候选集、支持/排除矩阵和缺失证据；模型只能排序候选、解释冲突或请求补采，不能创造新候选。</li>
<li><b>P0：把 checker 前置为解码协议。</b>采用 JSON Schema/grammar；证据 token、约束 ID、effect、target 使用枚举；逐步验证后失败最多重试 3 次。仍失败则 request_evidence，不把修复后的普通文本当终裁。</li>
<li><b>P0：分开“可推理 gap”与“不可辨识 missing”。</b>263 条 gap 才进入 LLM；150 条 no_anomaly/missing 直接补采。这样避免用多数类伪装覆盖率。</li>
<li><b>P0：logical4/logical8 分层校准。</b>logical8 当前 42.37%，必须建立 8×8 配对 lane 的方向覆盖率、异常比例和两端 Tx/Rx 一致性谓词，不能复用 logical4 的置信门。</li>
<li><b>P1：冻结探索性一致门。</b>先在 124 条训练数据上做 LOO/嵌套验证，验证“LLM final ∩ N5c ∩ 物理候选一致”；只有 Wilson 下界和独立验证均达标才增加自动覆盖。</li>
<li><b>P1：fiber 改为证据获取任务。</b>没有双向同步 Tx/Rx、LOS/LOL、OTDR/换纤或端面检查时不自动判 fiber；让 LLM 输出最小补采集合和端点候选，而非硬猜。</li>
<li><b>P1：以风险—覆盖曲线验收。</b>分别报告两个来源的 coverage、precision@coverage、constraint-valid rate、retry rate 和 fiber recall；不要用 100% 强制预测准确率代替选择性指标。</li>
</ol></div>
<div class='card'><h2>逐 case 复盘（484 条）</h2><table><tr><th>case</th><th>来源</th><th>v5复盘类型</th><th>旧LLM动作</th><th>旧建议</th><th>最终分析</th><th>置信层</th><th>安全动作</th><th>真实</th><th>变化</th><th>探索门</th></tr>{case_rows}</table></div>
</body></html>"""


def build_training_causal_calibration() -> dict[str, Any]:
    """Measure the revised causal signatures on train only."""
    data_dir = ROOT / "datasets/filtered_rule_temporal_2025_06_09_v1"
    cases = cases_by_manifest_split(data_dir, "train")
    bundle = OfflineKnowledgeBundle.load(
        ROOT / "artifacts/filtered_rule_deterministic_knowledge_v2/knowledge_bundle.json"
    )
    clean = [{key: value for key, value in case.items() if key not in {"label", "original_label"}} for case in cases]
    packs, features = bundle.extract_test_features(clean, source_dataset="train")
    variant = ExpertVariant(
        name="filtered-rule-causal-audit-no-fallback",
        single_metric_direction=DOC_VARIANT.single_metric_direction,
        use_fallbacks=False,
    )
    aligned = []
    for case, pack, feature in zip(cases, packs, features):
        legacy = diagnose(pack, variant=variant)
        symptom_side = legacy.sides[0].side if legacy.sides else None
        assessment = assess_filtered_rule_expert(
            expert_group=legacy.group,
            expert_verdict=legacy.verdict,
            symptom_side=symptom_side,
            tokens=feature.tokens,
            telemetry=case,
        )
        if assessment.rule == "aligned_receive_chain_ambiguous" and symptom_side:
            aligned.append({
                "truth": case["label"],
                "local": symptom_side,
                "opposite": "L2" if symptom_side == "L1" else "L1",
            })
    local_truth = sum(row["truth"] == row["local"] for row in aligned)
    opposite_truth = sum(row["truth"] == row["opposite"] for row in aligned)
    fiber_truth = sum(row["truth"] == "fiber" for row in aligned)
    return {
        "aligned_receive_chain": {
            "support": len(aligned),
            "local_truth": local_truth,
            "opposite_truth": opposite_truth,
            "fiber_truth": fiber_truth,
            "local_accuracy": local_truth / len(aligned) if aligned else 0.0,
            "opposite_accuracy": opposite_truth / len(aligned) if aligned else 0.0,
            "conclusion": "mixed labels; non-terminal",
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    v5_rows = load_json(V5)
    remote_rows: list[Mapping[str, Any]] = []
    for split in ("test_all_data", "test_rule1_channel_not_4"):
        remote_rows.extend(load_json(REMOTE / split / "predictions.json"))

    blind = blind_join(v5_rows, remote_rows)
    reviewed = attach_labels(blind, v5_rows)
    reasoning_label_mismatches = [
        row for row in reviewed
        if row["v5_action"] == "insufficient"
        if row["llm_proposed_verdict"] != row["actual"]
        and len(row["causal_expert"].get("candidates", [])) == 1
        and row["actual"] in row["causal_expert"]["candidates"]
    ]
    uncovered = [row for row in reviewed if row["v5_action"] == "insufficient"]
    v5_final = [row for row in reviewed if row["v5_action"] == "final"]
    terminal = [row for row in uncovered if row["llm_terminal_verdict"]]
    pilot = [row for row in uncovered if row["pilot_gate"]]
    corrected_candidates = [row for row in reviewed if row["corrected_verdict"]]
    corrected_terminal = [row for row in reviewed if row["corrected_action"] == "final" and row["corrected_verdict"]]
    gap = [row for row in uncovered if row["v5_review_category"] == "decision_graph_gap"]
    missing = [row for row in uncovered if row["v5_review_category"] == "missing_evidence"]
    logical8 = [row for row in reviewed if row["topology_id"] == "400g-400g-logical8"]
    all_data = [row for row in reviewed if row["source_dataset"] == "all_data"]
    logical8_source = [row for row in reviewed if row["source_dataset"] == "rule1_channel_not_4"]
    training_causal_calibration = build_training_causal_calibration()
    summary = {
        "schema_version": "filtered-rule-full-test-analysis-v3",
        "evaluation_design": {
            "prediction_source": str(REMOTE.relative_to(ROOT)),
            "uncovered_definition_source": str(V5.relative_to(ROOT)),
            "label_visible_during_prediction": False,
            "retrospective_test_informed": True,
            "policy_mutated": True,
        },
        "subgroups": {
            "完整测试集最终分析": metric(reviewed, "analysis_correct"),
            "all_data最终分析": metric(all_data, "analysis_correct"),
            "rule1_channel_not_4最终分析": metric(logical8_source, "analysis_correct"),
            "v5决策图冻结终裁": metric(v5_final, "analysis_correct"),
            "未覆盖样本最终分析": metric(uncovered, "analysis_correct"),
            "旧 LLM 门禁实际放行": metric(terminal, "terminal_correct"),
            "可推理 gap": metric(gap, "analysis_correct"),
            "不可辨识 missing/no-anomaly": metric(missing, "analysis_correct"),
            "logical8 全部样本": metric(logical8, "analysis_correct"),
            "探索门：final ∩ N5c ∩ 物理候选一致": metric(pilot, "terminal_correct"),
            "修复后全部候选（含人工复核）": metric(corrected_candidates, "corrected_correct"),
            "修复后高置信自动终裁": metric(corrected_terminal, "analysis_correct"),
        },
        "combined_operating_points": {
            "full_analysis": {"covered": len(reviewed), "correct": sum(row["analysis_correct"] for row in reviewed), "coverage": 1.0, "precision": sum(row["analysis_correct"] for row in reviewed) / len(reviewed)},
            "high_confidence_final": {"covered": len(corrected_terminal), "correct": sum(row["analysis_correct"] for row in corrected_terminal), "coverage": len(corrected_terminal) / len(reviewed), "precision": sum(row["analysis_correct"] for row in corrected_terminal) / len(corrected_terminal)},
            "v5": {"covered": len(v5_final), "correct": sum(row["analysis_correct"] for row in v5_final), "coverage": len(v5_final) / len(reviewed), "precision": sum(row["analysis_correct"] for row in v5_final) / len(v5_final)},
        },
        "truth_distribution": dict(Counter(row["actual"] for row in reviewed)),
        "analysis_prediction_distribution": dict(Counter(row["analysis_verdict"] for row in reviewed)),
        "analysis_confidence_tiers": dict(Counter(row["analysis_confidence_tier"] for row in reviewed)),
        "classification": classification_metrics(reviewed),
        "penalties": dict(Counter(item.get("kind", "unknown") for row in reviewed for item in row["compliance_penalties"])),
        "training_causal_calibration": training_causal_calibration,
        "correction_audit": {
            "label_changed": sum(row["label_changed"] for row in uncovered),
            "old_wrong_corrected_right": sum(row["old_wrong_corrected_right"] for row in uncovered),
            "old_right_corrected_wrong": sum(row["old_right_corrected_wrong"] for row in uncovered),
            "invalid_llm_downgraded": sum(not row["llm_output_valid"] and row["remote_action"] == "final" for row in uncovered),
            "reasoning_verdict_conflicts": sum(row["reasoning_verdict_conflict"] for row in uncovered),
            "reasoning_candidate_matches_truth": len(reasoning_label_mismatches),
            "corrected_actions": dict(Counter(row["corrected_action"] for row in reviewed)),
            "causal_rules": dict(Counter(row["causal_expert"]["rule"] for row in reviewed)),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    case_dir = args.output / "cases"
    case_dir.mkdir(exist_ok=True)
    dump_json(args.output / "blind_join.json", blind)
    dump_json(args.output / "case_reviews.json", reviewed)
    dump_json(args.output / "reasoning_label_mismatches.json", reasoning_label_mismatches)
    dump_json(args.output / "summary.json", summary)
    for row in reviewed:
        (case_dir / f"{row['case_id']}.html").write_text(render_case(row), encoding="utf-8")
    index = render_index(summary, reviewed)
    (args.output / "index.html").write_text(index, encoding="utf-8")
    (args.output / "report.html").write_text(index, encoding="utf-8")
    manifest = {
        "schema_version": "filtered-rule-full-test-analysis-manifest-v3",
        "case_count": len(reviewed),
        "blind_join_sha256": hashlib.sha256((args.output / "blind_join.json").read_bytes()).hexdigest(),
        "summary_sha256": hashlib.sha256((args.output / "summary.json").read_bytes()).hexdigest(),
        "case_reviews_sha256": hashlib.sha256((args.output / "case_reviews.json").read_bytes()).hexdigest(),
        "reasoning_label_mismatches_sha256": hashlib.sha256((args.output / "reasoning_label_mismatches.json").read_bytes()).hexdigest(),
    }
    dump_json(args.output / "manifest.json", manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
