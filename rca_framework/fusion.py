from __future__ import annotations

from typing import Any, Dict, List

from .types import CaseEvidence, ROOT_CAUSES, normalize_scores, rank_scores


def _top_evidence(method1: Dict[str, Any], method2: Dict[str, Any], label: str) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    for path in method1.get("graph_paths", []):
        if path.get("root_cause") == label:
            evidence.append({"source": "KG_RAG_LLM", "type": "graph_path", "detail": path})
        if len(evidence) >= 4:
            break
    for rule in method1.get("matched_kg_feature_rules", {}).get(label, [])[:4]:
        evidence.append({"source": "KG_FEATURE_RULE", "type": "class_feature_rule", "detail": rule})
    for rule in method2.get("matched_rules", {}).get(label, [])[:4]:
        evidence.append({"source": "KG_RCA", "type": "symbolic_rule", "detail": rule})
    return evidence


def _conflicting_evidence(
    method1: Dict[str, Any], method2: Dict[str, Any], prediction: str,
) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    for label in ROOT_CAUSES:
        if label == prediction:
            continue
        evidence.extend(_top_evidence(method1, method2, label))
    return evidence[:8]


def fuse_results(
    case: CaseEvidence,
    method1: Dict[str, Any],
    method2: Dict[str, Any],
    graph_weight: float = 0.55,
    symbolic_weight: float = 0.45,
    dominance_gap: float = 0.20,
    review_margin: float = 0.10,
) -> Dict[str, Any]:
    first_scores = normalize_scores(method1.get("scores", {}))
    second_scores = normalize_scores(method2.get("scores", {}))
    first_pred, second_pred = method1["prediction"], method2["prediction"]
    first_conf, second_conf = float(method1.get("confidence", 0.0)), float(method2.get("confidence", 0.0))
    blended = normalize_scores({
        label: graph_weight * first_scores[label] * (0.5 + first_conf) + symbolic_weight * second_scores[label] * (0.5 + second_conf)
        for label in ROOT_CAUSES
    })
    ranking = rank_scores(blended)
    margin = ranking[0][1] - ranking[1][1]

    if first_pred == second_pred:
        prediction = first_pred
        status = "agreement"
        rationale = "两条独立推理链结论一致；合并图路径与符号规则完成信息补全。"
    elif first_conf - second_conf >= dominance_gap:
        prediction = first_pred
        status = "conflict_resolved_by_kg_rag_llm"
        rationale = "两路结论冲突；KG+RAG+LLM 的校准置信度显著更高，采用其结论并保留规则反证。"
    elif second_conf - first_conf >= dominance_gap:
        prediction = second_pred
        status = "conflict_resolved_by_symbolic_rules"
        rationale = "两路结论冲突；符号规则的匹配置信度显著更高，采用其结论并保留路径反证。"
    else:
        prediction = ranking[0][0]
        status = "conflict_resolved_by_weighted_evidence" if margin >= review_margin else "manual_review_recommended"
        rationale = (
            "两路结论冲突且单路优势不足；按置信度、证据覆盖率和候选分差加权决策。"
            if margin >= review_margin else
            "两路结论冲突且融合分差过小；给出暂定三分类结果，同时建议人工复核。"
        )

    confidence = blended[prediction]
    if status == "agreement":
        confidence = min(1.0, 0.5 * confidence + 0.25 * first_conf + 0.25 * second_conf + 0.1)
    elif status == "manual_review_recommended":
        confidence = min(confidence, 0.5)
    missing = list(dict.fromkeys(case.missing_fields + method1.get("missing_information", [])))
    if not case.anomalies:
        missing.insert(0, "未提取到异常行为，请补充原始 lane 指标与 LOS/LOL 状态")
    return {
        "prediction": prediction,
        "confidence": round(confidence, 8),
        "decision_status": status,
        "method_predictions": {
            "KG_RAG_LLM": {"prediction": first_pred, "confidence": first_conf},
            "KG_RCA": {"prediction": second_pred, "confidence": second_conf},
        },
        "fused_scores": blended,
        "score_margin": round(margin, 8),
        "rationale": rationale,
        "supporting_evidence": _top_evidence(method1, method2, prediction),
        "conflicting_evidence": _conflicting_evidence(method1, method2, prediction),
        "information_completion": {
            "data_coverage": round(case.coverage, 8),
            "missing_or_requested_fields": missing,
            "explanation_sources": ["异常图路径", "训练集相似 case", "互斥符号规则"],
        },
    }
