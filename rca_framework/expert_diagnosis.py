"""Expert-informed diagnosis-path graphs for the expanded RCA training set.

The observable five-layer graph records *what was measured*.  This module adds
an auditable diagnosis layer that records how physical boundaries, expert
heuristics, competing hypotheses, exclusions, and missing evidence relate.

The implementation is deliberately label-free while extracting a case graph.
Labels are attached by the offline summarizer only after every evidence path has
been built.  This keeps the graph useful for retrieval without leaking the
training outcome into its signature.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from .expanded_evidence import (
    OPTICAL_DROP_BOUNDARY,
    OPTICAL_SENTINEL,
    SERDES_INVALID_BOUNDARY,
    SNR_INVALID_BOUNDARY,
    case_quality_state,
    is_exact_sentinel,
)


EXPERT_DIAGNOSIS_GRAPH_VERSION = "expanded-expert-diagnosis-graph-v1"
EXPERT_PATTERN_VERSION = "expanded-expert-patterns-v1"
SIDES = ("L1", "L2")
PEER = {"L1": "L2", "L2": "L1"}


@dataclass(frozen=True)
class ExpertPattern:
    pattern_id: str
    title: str
    source: str
    status: str
    meaning: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "pattern_id": self.pattern_id,
            "title": self.title,
            "source": self.source,
            "status": self.status,
            "meaning": self.meaning,
        }


EXPERT_PATTERNS: Tuple[ExpertPattern, ...] = (
    ExpertPattern(
        "EP_Q0_BLACKOUT", "全链路 blackout", "measurement-contract:M4",
        "veto", "双端收发光与 SNR 同时触底时优先解释为量测无效，不做端点归因。",
    ),
    ExpertPattern(
        "EP_TX_OFF", "本端发送器未出光", "physics:P1/P4/P5 + expert annotations",
        "physical", "bias=0 或 TX<=-39 表示本端发送链失效；精确 -40 单独保留为哨兵。",
    ),
    ExpertPattern(
        "EP_RX_HARD_DOWN", "接收方向硬失效", "expert annotations: repeated hard-RX pattern",
        "expert_supported", "同侧 RX<=-39且media_snr<=0确定该接收方向断；SerDes状态作协同证据，但不能单独区分对端发送、介质和本端接收。",
    ),
    ExpertPattern(
        "EP_DECODE_WITH_LIGHT", "有收光但解码链失效", "expert annotations: repeated RX-normal pattern",
        "expert_heuristic", "同侧 RX 有光而 media/SerDes 失效；专家标注多指向对端发送质量，但物理上也可能是本端解码链。",
    ),
    ExpertPattern(
        "EP_BILATERAL_RX_DEGRADATION", "双端接收侧同时劣化", "expert annotations + measurement-contract:M6",
        "candidate_only", "不用跨端 TX-RX 相减；双端各自出现接收劣化时仅提升 fiber 候选，并请求 OTDR/镜检。",
    ),
    ExpertPattern(
        "EP_INSUFFICIENT", "当前快照不可辨识", "expert annotations: missing_critical",
        "abstain", "没有硬边界或方向性证据时不根据告警端或多数类猜测。",
    ),
)


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def lane_values(case: Mapping[str, Any], metric: str, side: str) -> Dict[str, float]:
    """Return same-side lane readings while preserving lane identity."""
    value = case.get(metric)
    if not isinstance(value, Mapping):
        return {}
    side_value = value.get(side)
    if not isinstance(side_value, Mapping):
        return {}
    result = {}
    for lane, raw in side_value.items():
        parsed = _number(raw)
        if parsed is not None:
            result[str(lane)] = parsed
    return result


def classify_reading(metric: str, value: float) -> str:
    """Metric-specific state; zero is intentionally *not* a global sentinel."""
    if metric in {"txpower", "rxpower"}:
        if is_exact_sentinel(value, OPTICAL_SENTINEL):
            return "exact_minus_40_sentinel"
        if value <= OPTICAL_DROP_BOUNDARY:
            return "no_light"
        return "light_present"  # 0 dBm belongs here.
    if metric in {"media_snr", "host_snr"}:
        return "invalid_or_floor" if value <= SNR_INVALID_BOUNDARY else "valid"
    if metric == "serdes_snr":
        return "invalid_state" if value <= SERDES_INVALID_BOUNDARY else "valid_state"
    if metric == "bias":
        return "laser_not_driven" if math.isclose(value, 0.0, abs_tol=1e-9) else "laser_driven"
    raise KeyError(metric)


def _node(node_id: str, node_type: str, **attrs: Any) -> Dict[str, Any]:
    return {"id": node_id, "type": node_type, "attrs": attrs}


def _edge(src: str, dst: str, edge_type: str, **attrs: Any) -> Dict[str, Any]:
    return {"src": src, "dst": dst, "type": edge_type, "attrs": attrs}


def _add_node(nodes: Dict[str, Dict[str, Any]], node_id: str, node_type: str, **attrs: Any) -> str:
    nodes.setdefault(node_id, _node(node_id, node_type, **attrs))
    return node_id


def _add_candidate(
    nodes: Dict[str, Dict[str, Any]], edges: list[Dict[str, Any]], hypothesis: str,
    candidate: str, effect: str, *, reason: str,
) -> None:
    candidate_id = _add_node(nodes, f"candidate:{candidate}", "CandidateRootCause", label=candidate)
    edges.append(_edge(hypothesis, candidate_id, effect, reason=reason))


def _peer_tx_present(case: Mapping[str, Any], side: str) -> bool:
    values = lane_values(case, "txpower", PEER[side]).values()
    return bool(values) and any(classify_reading("txpower", value) == "light_present" for value in values)


def _side_scalar(case: Mapping[str, Any], field: str, side: str) -> Any:
    value = case.get(field)
    return value.get(side) if isinstance(value, Mapping) else value


def build_expert_diagnosis_graph(case: Mapping[str, Any]) -> Dict[str, Any]:
    """Build one label-free expert diagnosis graph from raw telemetry."""
    case_id = str(case.get("case_id", ""))
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: list[Dict[str, Any]] = []
    matched_patterns: set[str] = set()
    missing_evidence: set[str] = set()
    _add_node(nodes, "case", "Case", case_id=case_id)
    sop_steps = (
        ("step:Q0", "Q0", "先验证缺失、blackout与哨兵可信度"),
        ("step:P_TX", "P", "检查bias=0、TX=-40/<=-39与本端发送链"),
        ("step:R_RX", "R", "检查RX、media/host SNR、SerDes及端间方向关系"),
        ("step:F_MEDIUM", "F", "检查双端接收劣化与fiber补采条件"),
        ("step:D", "D", "汇总候选、排除、竞争关系和缺失证据"),
    )
    previous_step = ""
    for step_id, order, statement in sop_steps:
        _add_node(nodes, step_id, "SOPStep", order=order, statement=statement)
        edges.append(_edge("case", step_id, "HAS_STEP"))
        if previous_step:
            edges.append(_edge(previous_step, step_id, "PRECEDES"))
        previous_step = step_id
    context_id = _add_node(
        nodes, "context:alarm", "AlarmContext",
        alarm_name=case.get("alarm_name"), alarm_ip_interface=case.get("alarm_ip_interface"),
        physical_meaning="触发与端口上下文；不作为根因物理证据",
    )
    edges.append(_edge("case", context_id, "HAS_CONTEXT"))

    quality = case_quality_state(dict(case))
    quality_id = _add_node(
        nodes, f"quality:{quality['quality']}", "QualityState",
        quality=quality["quality"], missing=list(quality["missing_measurements"]),
    )
    edges.append(_edge("case", quality_id, "HAS_QUALITY_STATE"))
    if quality["optical_blackout"]:
        matched_patterns.add("EP_Q0_BLACKOUT")
        pattern_id = _add_node(nodes, "pattern:EP_Q0_BLACKOUT", "ExpertPattern", status="veto")
        edges.append(_edge(quality_id, pattern_id, "TRIGGERS"))
        edges.append(_edge("step:Q0", pattern_id, "EVALUATES"))
        review = _add_node(nodes, "action:human_review", "DecisionAction", action="human_review")
        edges.append(_edge(pattern_id, review, "REQUIRES"))
        return _finalize(case_id, nodes, edges, matched_patterns, {"现场确认模块与采集链路"})

    # Q0 missingness is a gate and never positive root-cause evidence.
    for item in quality["missing_measurements"]:
        missing_id = _add_node(nodes, f"missing:{item}", "MissingEvidence", measurement=item)
        edges.append(_edge(quality_id, missing_id, "LACKS"))
        missing_evidence.add(item)

    side_rx_degraded: Dict[str, bool] = {side: False for side in SIDES}
    side_has_directional_pattern: Dict[str, bool] = {side: False for side in SIDES}
    for side in SIDES:
        peer = PEER[side]
        for field, unit, physical_meaning in (
            ("Temperature", "degC", "模块环境/热状态；0-70 degC规格内只作排除证据"),
            ("Voltage", "V", "模块供电状态；3.3 V ±5%外只提升本端设备候选"),
        ):
            raw = _number(_side_scalar(case, field, side))
            if raw is None:
                continue
            state = (
                "within_spec" if field == "Temperature" and 0.0 <= raw <= 70.0
                else "within_spec" if field == "Voltage" and 3.135 <= raw <= 3.465
                else "outside_spec"
            )
            scalar_id = _add_node(
                nodes, f"measurement:{side}:{field}", "MeasurementState", side=side, metric=field,
                unit=unit, value=raw, state=state, physical_meaning=physical_meaning,
            )
            edges.append(_edge("case", scalar_id, "OBSERVES"))
        for status_field in ("RxLOS", "RxLOL", "TxLOS", "TxLOL"):
            raw_status = _side_scalar(case, status_field, side)
            if raw_status is None:
                continue
            status_id = _add_node(
                nodes, f"measurement:{side}:{status_field}", "MeasurementState", side=side,
                metric=status_field, value=str(raw_status),
                path="receive" if status_field.startswith("Rx") else "transmit",
                physical_meaning=(
                    "本端接收侧失光/失锁症状；根因链包含对端发送、介质和本端接收"
                    if status_field.startswith("Rx") else "本端发送侧失光/失锁症状"
                ),
            )
            edges.append(_edge("case", status_id, "OBSERVES"))
        lane_number = _side_scalar(case, "Lane number", side)
        if lane_number is not None:
            lane_id = _add_node(
                nodes, f"topology:{side}:lane_number", "TopologyContext", side=side,
                lane_number=lane_number, physical_meaning="告警/降lane上下文，不等于遥测数组宽度",
            )
            edges.append(_edge("case", lane_id, "HAS_CONTEXT"))
        # Preserve every measured feature and its physical state even when it is
        # normal.  Normal/exclusion evidence is essential to a diagnosis chain;
        # it is not part of the root-cause signature by itself.
        for metric in ("bias", "txpower", "rxpower", "media_snr", "host_snr", "serdes_snr"):
            readings = lane_values(case, metric, side)
            if not readings:
                continue
            states = Counter(classify_reading(metric, value) for value in readings.values())
            measurement_id = _add_node(
                nodes, f"measurement:{side}:{metric}", "MeasurementState", side=side, metric=metric,
                unit={
                    "bias": "mA", "txpower": "dBm", "rxpower": "dBm",
                    "media_snr": "field-unit", "host_snr": "field-unit",
                    "serdes_snr": "unknown/binary-validity-only",
                }[metric],
                lane_count=len(readings), minimum=min(readings.values()), maximum=max(readings.values()),
                spread=max(readings.values()) - min(readings.values()), states=dict(states),
                zero_semantics=(
                    "normal optical level when observed" if metric in {"txpower", "rxpower"}
                    else "invalid/floor" if metric in {"media_snr", "host_snr"}
                    else "invalid state at <=1" if metric == "serdes_snr"
                    else "laser not driven"
                ),
            )
            edges.append(_edge("case", measurement_id, "OBSERVES"))
        # P: a zero bias or optical TX drop is local-send evidence.
        tx = lane_values(case, "txpower", side)
        bias = lane_values(case, "bias", side)
        off_lanes = {
            lane for lane, value in tx.items()
            if classify_reading("txpower", value) in {"exact_minus_40_sentinel", "no_light"}
        } | {
            lane for lane, value in bias.items() if classify_reading("bias", value) == "laser_not_driven"
        }
        for lane in sorted(off_lanes):
            evidence_id = _add_node(
                nodes, f"evidence:{side}:tx_off:{lane}", "BoundaryEvidence", side=side, lane=lane,
                predicate="bias == 0 OR txpower <= -39 dBm", source="P1/P4/P5",
                txpower=tx.get(lane), bias=bias.get(lane),
            )
            edges.append(_edge("case", evidence_id, "OBSERVES"))
            pattern_id = _add_node(nodes, f"pattern:EP_TX_OFF:{side}", "ExpertPattern", side=side)
            edges.append(_edge(evidence_id, pattern_id, "SATISFIES"))
            edges.append(_edge("step:P_TX", pattern_id, "EVALUATES"))
            _add_candidate(nodes, edges, pattern_id, side, "SUPPORTS", reason="本端发送器没有出光")
            _add_candidate(nodes, edges, pattern_id, "fiber", "EXCLUDES", reason="介质不能解释未发出的光")
            matched_patterns.add("EP_TX_OFF")
            side_has_directional_pattern[side] = True

        rx = lane_values(case, "rxpower", side)
        media = lane_values(case, "media_snr", side)
        serdes = lane_values(case, "serdes_snr", side)
        common_lanes = sorted(set(rx) & set(media) & set(serdes))
        for lane in common_lanes:
            rx_state = classify_reading("rxpower", rx[lane])
            media_bad = classify_reading("media_snr", media[lane]) == "invalid_or_floor"
            serdes_bad = classify_reading("serdes_snr", serdes[lane]) == "invalid_state"
            rx_down = rx_state in {"exact_minus_40_sentinel", "no_light"}
            if not (media_bad and (rx_down or serdes_bad)):
                continue
            pattern = "EP_RX_HARD_DOWN" if rx_down else "EP_DECODE_WITH_LIGHT"
            evidence_id = _add_node(
                nodes, f"evidence:{side}:{pattern}:{lane}", "BoundaryEvidence", side=side, lane=lane,
                rxpower=rx[lane], media_snr=media[lane], serdes_snr=serdes[lane],
                predicates=[
                    "rxpower <= -39" if rx_down else "rxpower > -39", "media_snr <= 0",
                    "serdes_snr <= 1" if serdes_bad else "serdes_snr observed above hard-failure sentinel",
                ],
                peer_tx_present=_peer_tx_present(case, side),
            )
            edges.append(_edge("case", evidence_id, "OBSERVES"))
            pattern_id = _add_node(nodes, f"pattern:{pattern}:{side}", "ExpertPattern", side=side, peer=peer)
            edges.append(_edge(evidence_id, pattern_id, "SATISFIES"))
            edges.append(_edge("step:R_RX", pattern_id, "EVALUATES"))
            if rx_down:
                _add_candidate(nodes, edges, pattern_id, side, "SUPPORTS", reason="专家重复模式支持本端接收链")
                _add_candidate(nodes, edges, pattern_id, "fiber", "COMPETES_WITH", reason="无现场介质证据，不能排除传输路径")
                _add_candidate(nodes, edges, pattern_id, peer, "COMPETES_WITH", reason="对端TX有数值只证明有光，不证明发送信号质量正常")
            else:
                _add_candidate(nodes, edges, pattern_id, peer, "SUPPORTS", reason="专家标注中的对端发送质量启发式")
                _add_candidate(nodes, edges, pattern_id, side, "COMPETES_WITH", reason="本端解码/电链同样可产生该观测")
                _add_candidate(nodes, edges, pattern_id, "fiber", "COMPETES_WITH", reason="色散、反射或串扰可在收光存在时破坏信号质量")
            matched_patterns.add(pattern)
            side_rx_degraded[side] = True
            side_has_directional_pattern[side] = True

    if all(side_rx_degraded.values()) and "EP_TX_OFF" not in matched_patterns:
        matched_patterns.add("EP_BILATERAL_RX_DEGRADATION")
        pattern_id = _add_node(nodes, "pattern:EP_BILATERAL_RX_DEGRADATION", "ExpertPattern", status="candidate_only")
        edges.append(_edge("case", pattern_id, "SATISFIES"))
        edges.append(_edge("step:F_MEDIUM", pattern_id, "EVALUATES"))
        _add_candidate(nodes, edges, pattern_id, "fiber", "SUPPORTS", reason="双端各自存在接收劣化，但未计算跨端绝对损耗")
        request = _add_node(nodes, "action:request_otdr", "DecisionAction", action="request_evidence")
        edges.append(_edge(pattern_id, request, "REQUIRES", evidence=["OTDR", "端面镜检", "换纤复测"]))

    if not any(side_has_directional_pattern.values()) and "EP_BILATERAL_RX_DEGRADATION" not in matched_patterns:
        matched_patterns.add("EP_INSUFFICIENT")
        pattern_id = _add_node(nodes, "pattern:EP_INSUFFICIENT", "ExpertPattern", status="abstain")
        edges.append(_edge("case", pattern_id, "SATISFIES"))
        edges.append(_edge("step:D", pattern_id, "EVALUATES"))
        action = _add_node(nodes, "action:request_evidence", "DecisionAction", action="request_evidence")
        edges.append(_edge(pattern_id, action, "REQUIRES"))

    return _finalize(case_id, nodes, edges, matched_patterns, missing_evidence)


def _finalize(
    case_id: str, nodes: Mapping[str, Dict[str, Any]], edges: Sequence[Dict[str, Any]],
    matched_patterns: Iterable[str], missing_evidence: Iterable[str],
) -> Dict[str, Any]:
    ordered_nodes = [nodes[key] for key in sorted(nodes)]
    ordered_edges = sorted(edges, key=lambda item: (item["src"], item["type"], item["dst"]))
    signature = tuple(sorted(
        f"{edge['src']}|{edge['type']}|{edge['dst']}"
        for edge in ordered_edges
        if edge["type"] in {"SATISFIES", "SUPPORTS", "EXCLUDES", "COMPETES_WITH", "REQUIRES"}
    ))
    payload = {
        "schema_version": EXPERT_DIAGNOSIS_GRAPH_VERSION,
        "case_id": case_id,
        "nodes": ordered_nodes,
        "edges": ordered_edges,
        "matched_patterns": sorted(set(matched_patterns)),
        "missing_evidence": sorted(set(missing_evidence)),
        "diagnostic_signature": list(signature),
    }
    payload["content_hash"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return payload


def annotation_pattern_audit(annotation_document: Mapping[str, Any]) -> Dict[str, Any]:
    """Summarize expert prose without treating unsafe loss claims as constraints."""
    annotations = list(annotation_document.get("annotations", annotation_document.get("pairs", ())))
    unsafe = []
    safe = []
    for item in annotations:
        note = str(item.get("notes", ""))
        row = {
            "pattern_id": item.get("pattern_id"), "left_case_id": item.get("left_case_id"),
            "right_case_id": item.get("right_case_id"), "notes": note,
        }
        if re.search(r"TX\s*-\s*RX|光衰", note, flags=re.IGNORECASE):
            row["reason"] = "跨端 lane 配对/功率标定未确认，禁止自动沉淀绝对损耗约束"
            unsafe.append(row)
        else:
            safe.append(row)
    return {
        "schema_version": EXPERT_PATTERN_VERSION,
        "annotation_count": len(annotations),
        "safe_for_pattern_mining_count": len(safe),
        "requires_domain_confirmation_count": len(unsafe),
        "requires_domain_confirmation": unsafe,
        "patterns": [item.to_dict() for item in EXPERT_PATTERNS],
    }


def review_training_case(
    case: Mapping[str, Any], graph: Mapping[str, Any], *, label_status: str = "unreviewed",
    unsafe_expert_reasoning: bool = False,
) -> Dict[str, Any]:
    """Compare a finished label-free graph with its training label.

    This function never changes the graph or the label.  It identifies whether
    the snapshot actually narrows the candidate set enough to audit that label.
    """
    label = str(case.get("label", ""))
    support: Dict[str, list[str]] = {side: [] for side in (*SIDES, "fiber")}
    compete: Dict[str, list[str]] = {side: [] for side in (*SIDES, "fiber")}
    excluded: Dict[str, list[str]] = {side: [] for side in (*SIDES, "fiber")}
    for edge in graph.get("edges", ()):
        if not str(edge.get("dst", "")).startswith("candidate:"):
            continue
        candidate = str(edge["dst"]).split(":", 1)[1]
        reason = str(edge.get("attrs", {}).get("reason", ""))
        if edge["type"] == "SUPPORTS":
            support[candidate].append(reason)
        elif edge["type"] == "COMPETES_WITH":
            compete[candidate].append(reason)
        elif edge["type"] == "EXCLUDES":
            excluded[candidate].append(reason)

    tx_off_sides = sorted(
        side for side in SIDES if any(
            node["id"] == f"pattern:EP_TX_OFF:{side}" for node in graph.get("nodes", ())
        )
    )
    directional = bool({"EP_RX_HARD_DOWN", "EP_DECODE_WITH_LIGHT"} & set(graph.get("matched_patterns", ())))
    if len(tx_off_sides) == 1:
        candidates = (tx_off_sides[0],)
        review_class = "decisive_local_tx_off"
        rationale = "存在唯一一侧bias=0或TX<=-39；发送器未出光是比接收症状更强的本端证据"
    elif len(tx_off_sides) > 1:
        candidates = tuple(tx_off_sides)
        review_class = "conflicting_tx_off"
        rationale = "两侧都出现发送失效，当前快照不能压缩为单一根因"
    elif directional:
        candidates = tuple(
            candidate for candidate in ("L1", "L2", "fiber")
            if support[candidate] or compete[candidate]
        )
        review_class = "direction_observed_root_not_identifiable"
        rationale = "RX/SNR/SerDes确定了异常方向，但对端发送质量、介质和本端接收/解码链仍是竞争候选"
    else:
        candidates = ("L1", "L2", "fiber")
        review_class = "insufficient_snapshot"
        rationale = "没有硬发送边界或可唯一定位的第二条证据"

    if review_class == "decisive_local_tx_off":
        label_assessment = (
            "consistent_with_decisive_evidence" if label in candidates else "suspected_label_conflict"
        )
    elif label == "fiber":
        label_assessment = "fiber_label_requires_external_evidence"
    elif label in candidates:
        label_assessment = "compatible_but_not_identifiable"
    else:
        label_assessment = "not_explained_by_current_evidence"
    required = []
    if review_class == "direction_observed_root_not_identifiable":
        required.extend(("对端发送信号质量/激光器自检", "本端接收器与解码链自检"))
        if "fiber" in candidates:
            required.extend(("OTDR", "端面镜检或换纤复测"))
    if review_class in {"conflicting_tx_off", "insufficient_snapshot"}:
        required.append("同步重采双端遥测并核对告警时刻")
    if label == "fiber":
        required.extend(("OTDR", "端面镜检或换纤复测", "可信的同步双向功率标定"))
    return {
        "case_id": str(case.get("case_id", "")), "label": label, "label_status": label_status,
        "review_class": review_class, "candidate_set": list(candidates),
        "label_assessment": label_assessment, "rationale": rationale,
        "supporting_evidence": {key: value for key, value in support.items() if value},
        "competing_evidence": {key: value for key, value in compete.items() if value},
        "excluded_candidates": {key: value for key, value in excluded.items() if value},
        "required_evidence": list(dict.fromkeys(required)),
        "unsafe_expert_reasoning": unsafe_expert_reasoning,
        "unsafe_reasoning_note": (
            "专家说明涉及跨端TX-RX绝对损耗；当前lane对应/标定不成立，只保留审核记录"
            if unsafe_expert_reasoning else ""
        ),
    }


def summarize_training_graphs(
    cases: Sequence[Mapping[str, Any]], graphs: Sequence[Mapping[str, Any]],
    label_status: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    status = label_status or {}
    pattern_labels: Dict[str, Counter[str]] = {}
    pattern_side_labels: Dict[str, Counter[str]] = {}
    pattern_side_status_labels: Dict[Tuple[str, str], Counter[str]] = {}
    signatures: Dict[Tuple[str, ...], list[str]] = {}
    for case, graph in zip(cases, graphs):
        label = str(case.get("label", ""))
        for pattern in graph["matched_patterns"]:
            pattern_labels.setdefault(pattern, Counter())[label] += 1
        for node in graph["nodes"]:
            if node["type"] != "ExpertPattern":
                continue
            key = str(node["id"]).removeprefix("pattern:")
            review_status = status.get(str(case.get("case_id")), "unreviewed")
            pattern_side_labels.setdefault(key, Counter())[label] += 1
            pattern_side_status_labels.setdefault((key, review_status), Counter())[label] += 1
        signatures.setdefault(tuple(graph["diagnostic_signature"]), []).append(label)
    pattern_rows = []
    for pattern in EXPERT_PATTERNS:
        counts = pattern_labels.get(pattern.pattern_id, Counter())
        support = sum(counts.values())
        majority = counts.most_common(1)[0] if counts else ("", 0)
        pattern_rows.append({
            **pattern.to_dict(), "support": support, "label_distribution": dict(counts),
            "majority_label": majority[0], "majority_purity": round(majority[1] / support, 6) if support else None,
        })
    mixed = [labels for labels in signatures.values() if len(set(labels)) > 1]
    pattern_side_rows = []
    for key in sorted(pattern_side_labels):
        counts = pattern_side_labels[key]
        support = sum(counts.values())
        majority = counts.most_common(1)[0]
        status_rows = {
            review_status: dict(pattern_side_status_labels.get((key, review_status), Counter()))
            for review_status in ("expert_reviewed", "unreviewed")
        }
        pattern_side_rows.append({
            "pattern": key, "support": support, "label_distribution": dict(counts),
            "majority_label": majority[0], "majority_purity": round(majority[1] / support, 6),
            "by_label_status": status_rows,
        })
    return {
        "schema_version": EXPERT_DIAGNOSIS_GRAPH_VERSION,
        "case_count": len(cases),
        "label_distribution": dict(Counter(str(case.get("label", "")) for case in cases)),
        "label_status_distribution": dict(Counter(status.get(str(case.get("case_id")), "unreviewed") for case in cases)),
        "unique_diagnostic_signatures": len(signatures),
        "mixed_label_signature_groups": len(mixed),
        "cases_in_mixed_label_signatures": sum(len(items) for items in mixed),
        "pattern_summary": pattern_rows,
        "pattern_side_summary": pattern_side_rows,
        "boundary_semantics": {
            "bias == 0 mA": "激光器未驱动/发送链失效",
            "media_snr or host_snr <= 0": "无效或触底",
            "serdes_snr <= 1": "失效状态；量纲未知",
            "txpower or rxpower == -40 dBm": "精确哨兵；在非 blackout case 中表示无光",
            "txpower or rxpower <= -39 dBm": "工程断光区间",
            "txpower or rxpower == 0 dBm": "正常有光读数，不是故障边界",
        },
    }
