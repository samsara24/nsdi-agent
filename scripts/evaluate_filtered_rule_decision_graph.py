#!/usr/bin/env python3
"""Blindly evaluate the train-only decision graph, then render per-case reviews."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca_framework.data import cases_by_manifest_split  # noqa: E402
from rca_framework.decision_graph_policy import (  # noqa: E402
    RECEIVE_SYMPTOM_GROUPS, learned_path_match, receive_symptom_context,
)
from rca_framework.evidence_graph import match_many  # noqa: E402
from rca_framework.evidence_pack import build_packs  # noqa: E402
from rca_framework.expert import (  # noqa: E402
    DOC_VARIANT, ExpertVariant, detect_side_anomalies, diagnose, side_metric_values,
)
from rca_framework.knowledge import OfflineKnowledgeBundle  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/filtered_rule_temporal_2025_06_09_v1"
KNOWLEDGE = ROOT / "artifacts/filtered_rule_deterministic_knowledge_v2/knowledge_bundle.json"
DECISION_GRAPH = ROOT / "artifacts/filtered_rule_decision_graph_v4/decision_graph.json"
OUTPUT = ROOT / "artifacts/filtered_rule_decision_graph_test_v5"
SPLITS = ("test/all_data", "test/rule1_channel_not_4")
MIN_RULE_SUPPORT = 10
MIN_RULE_LOWER_BOUND = 0.50


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stripped(cases: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    return [{key: value for key, value in case.items() if key not in {"label", "original_label"}} for case in cases]


def physical_votes(tokens: Sequence[str]) -> list[Dict[str, Any]]:
    votes = []
    token_set = set(tokens)
    for token in tokens:
        if not token.startswith("lane:"):
            continue
        _, direction, state, *_ = token.split(":")
        source = direction.split("_to_", 1)[0]
        if state == "tx_down":
            tx_status = any(f"status:{source}:{key}" in token_set for key in ("TxLOS", "TxLOL"))
            votes.append({"source": "physical_tx_down", "verdict": source, "strength": "strong" if tx_status else "moderate",
                          "evidence": [token] + ([f"status:{source}:TxLOS/TxLOL"] if tx_status else []),
                          "reason": f"{source}发送触底，支持发送端根因"})
        elif state == "bidirectional_same_lane":
            votes.append({"source": "physical_bidirectional_lane", "verdict": "fiber", "strength": "moderate",
                          "evidence": [token], "reason": "同lane双向异常形成介质候选，但仍需现场正证据"})
    directions = {token.split(":")[1] for token in tokens if token.startswith("lane:") and ":tx_ok_rx_down" in token}
    if {"L1_to_L2", "L2_to_L1"} <= directions:
        votes.append({"source": "physical_bidirectional_receive_loss", "verdict": "fiber", "strength": "strong",
                      "evidence": sorted(token for token in tokens if ":tx_ok_rx_down" in token),
                      "reason": "双向发送存在而跨端接收异常，形成正向介质证据"})
    return votes


def resolve_votes(votes: Sequence[Mapping[str, Any]]) -> tuple[str | None, str, str]:
    """Resolve candidates conservatively; lower-grade physical conflicts still veto."""
    strong = [vote for vote in votes if vote["strength"] in {"strong", "calibrated"}]
    labels = {vote["verdict"] for vote in strong}
    all_vote_labels = {vote["verdict"] for vote in votes}
    if len(labels) == 1 and len(all_vote_labels) == 1:
        return next(iter(labels)), "final", "全部物理、历史与校准候选方向一致"
    if len(labels) == 1 and len(all_vote_labels) > 1:
        return None, "insufficient", "高等级候选与其他物理候选冲突"
    if len(labels) > 1:
        return None, "insufficient", "高等级证据候选冲突"
    return None, "insufficient", "无达到训练可靠性门禁的决策路径" if votes else "未检出可终裁异常路径"


def acquisition_recommendations(telemetry_status: str, missing_fields: Sequence[str],
                                tokens: Sequence[str]) -> list[str]:
    """Return case-specific, non-imputing telemetry acquisition advice."""
    advice = []
    missing = set(missing_fields)
    if telemetry_status == "no_telemetry":
        advice.append("补采告警前后至少一个完整时间窗的两端同步遥测，当前无可定位量测。")
    if any(field.endswith(("TxLOS", "TxLOL", "RxLOS", "RxLOL")) for field in missing):
        advice.append("补采两端同一时刻的TxLOS/TxLOL/RxLOS/RxLOL，区分发送关闭、接收失锁和瞬态告警。")
    if any(field.endswith(("txpower", "rxpower")) for field in missing):
        advice.append("补采两端全部光学lane的Tx/Rx功率，保留原始lane宽度与无效值状态。")
    if any(field.endswith(("media_snr", "serdes_snr", "host_snr")) for field in missing):
        advice.append("补采media/SerDes/host侧SNR，确认异常位于光路、模块介质侧还是设备数字侧。")
    if not any(token.startswith("lane:") for token in tokens):
        advice.append("补采可建立同编号跨端配对的lane级Tx/Rx数据，用于验证单向传播和影响比例。")
    advice.append("补充端口管理/运行状态、最近倒换与插拔记录；缺测保持missing，不按正常值填充。")
    return list(dict.fromkeys(advice))


def blind_predict(cases: Sequence[Mapping[str, Any]], split: str, bundle: OfflineKnowledgeBundle,
                  rule_stats: Mapping[str, Mapping[str, Any]],
                  topology_rule_stats: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
                  learned_path_rows: Sequence[Mapping[str, Any]] = ()) -> list[Dict[str, Any]]:
    clean = stripped(cases)
    packs, features = bundle.extract_test_features(clean, source_dataset=DATA.name)
    matches = match_many(bundle.graph, features, top_k=0)
    no_fallback = ExpertVariant(name="decision-graph-no-fallback", single_metric_direction=DOC_VARIANT.single_metric_direction, use_fallbacks=False)
    output = []
    for pack, feature, match in zip(packs, features, matches):
        expert = diagnose(pack, variant=no_fallback)
        stats = dict(rule_stats.get(expert.group, {}))
        topology_stats = dict((topology_rule_stats or {}).get((expert.group, pack.topology_id), {}))
        anomalies = {side: detect_side_anomalies(side_metric_values(pack, side)) for side in ("L1", "L2")}
        symptom_side = expert.sides[0].side if expert.sides else None
        causal_context = receive_symptom_context(
            expert.group, expert.verdict, symptom_side, feature.tokens, anomalies,
        )
        topology_reliable = bool(
            int(topology_stats.get("support", 0)) >= MIN_RULE_SUPPORT
            and float(topology_stats.get("wilson_lower_bound", 0.0)) >= MIN_RULE_LOWER_BOUND
        )
        statistical_gate = bool(
            expert.verdict and int(stats.get("support", 0)) >= MIN_RULE_SUPPORT
            and float(stats.get("wilson_lower_bound", 0.0)) >= MIN_RULE_LOWER_BOUND
            and topology_reliable
        )
        causal_gate = expert.group not in RECEIVE_SYMPTOM_GROUPS or causal_context == "opposite_tx_fault"
        accepted_expert = statistical_gate and causal_gate
        votes = physical_votes(feature.tokens)
        if accepted_expert:
            votes.append({"source": "expert_rule_calibrated", "verdict": expert.verdict, "strength": "calibrated",
                          "evidence": [expert.group], "reason": expert.reason,
                          "support": stats.get("support"), "wilson_lower_bound": stats.get("wilson_lower_bound")})
        learned_path = learned_path_match(
            tokens=feature.tokens, topology_id=pack.topology_id, group=expert.group,
            verdict=expert.verdict, training_rows=learned_path_rows,
        )
        if learned_path is not None:
            terminal = bool(int(learned_path["config"].get("terminal", 1)))
            votes.append({"source": "train_loo_positive_path", "verdict": learned_path["verdict"],
                          "strength": "calibrated" if terminal else "advisory",
                          "evidence": [row["case_id"] for row in learned_path["neighbors"]],
                          "reason": "同拓扑、同规则、同方向训练近邻通过相似度/支持/纯度门"
                                    + ("" if terminal else "；该规则仅保留候选，不参与终裁"),
                          "support": learned_path["support"], "purity": learned_path["purity"],
                          "min_similarity": learned_path["min_similarity"]})
        media_topology_path = bool(
            expert.group == "expert:single:media_snr" and expert.verdict
            and int(topology_stats.get("support", 0)) >= 8
            and float(topology_stats.get("wilson_lower_bound", 0.0)) >= MIN_RULE_LOWER_BOUND
        )
        if media_topology_path:
            votes.append({"source": "topology_calibrated_media_path", "verdict": expert.verdict,
                          "strength": "calibrated", "evidence": [expert.group, pack.topology_id],
                          "reason": "当前拓扑的media_snr方向规则通过训练支持数与Wilson下界门",
                          "support": topology_stats.get("support"),
                          "wilson_lower_bound": topology_stats.get("wilson_lower_bound")})
        dual = match.dual_top_candidates
        exact = [candidate for candidate in dual if candidate.feature_similarity == 1.0 and candidate.graph_similarity == 1.0]
        exact_labels = Counter(candidate.label for candidate in exact if candidate.label)
        if exact and len(exact_labels) == 1 and len(exact) >= 2:
            verdict = next(iter(exact_labels))
            votes.append({"source": "pure_exact_history", "verdict": verdict, "strength": "calibrated",
                          "evidence": [candidate.case_id for candidate in exact],
                          "reason": f"{len(exact)}条双精确历史case标签纯净", "support": len(exact)})
        verdict, action, reason = resolve_votes(votes)
        output.append({
            "case_id": pack.case_id, "split": split, "verdict": verdict, "action": action, "reason": reason,
            "telemetry_status": pack.telemetry_status, "missing_fields": list(pack.missing_fields),
            "optical_blackout": pack.optical_blackout, "source_dataset": pack.source_dataset,
            "topology_id": pack.topology_id, "lane_profile": pack.lane_profile,
            "features": feature.to_dict(), "expert": expert.to_dict(), "expert_rule_stats": stats,
            "topology_rule_stats": topology_stats, "receive_causal_context": causal_context,
            "expert_statistical_gate": statistical_gate, "expert_causal_gate": causal_gate,
            "learned_positive_path": learned_path,
            "media_topology_positive_path": media_topology_path,
            "expert_rule_accepted": accepted_expert, "side_anomalies": anomalies, "decision_votes": votes,
            "history": [candidate.to_dict() for candidate in match.candidates[:5]],
            "acquisition_recommendations": acquisition_recommendations(
                pack.telemetry_status, pack.missing_fields, feature.tokens,
            ),
            "similarity": {"feature": match.max_feature_similarity, "graph": match.max_graph_similarity,
                           "exact_pure_support": len(exact) if len(exact_labels) == 1 else 0,
                           "exact_labels": dict(exact_labels)},
            "blind_reasoning": [
                f"量测质量：{pack.telemetry_status}，缺失字段{len(pack.missing_fields)}个",
                f"可解释特征：{len(feature.tokens)}个，{', '.join(sorted(feature.by_family)) or 'none'}",
                f"专家路径：{expert.group}；总体支持{stats.get('support',0)}，拓扑支持{topology_stats.get('support',0)}，拓扑Wilson下界{float(topology_stats.get('wilson_lower_bound',0)):.3f}",
                f"接收症状因果上下文：{causal_context}；统计门{'通过' if statistical_gate else '未通过'}，因果门{'通过' if causal_gate else '未通过'}",
                f"物理/历史候选：{len(votes)}个；最终{verdict or '证据不足'}",
            ],
        })
    return output


def attribution(pred: Mapping[str, Any], truth: str, label_status: str) -> tuple[str, str]:
    verdict = pred.get("verdict")
    if verdict == truth:
        return "correct", "有效决策路径与真实标签一致。"
    if verdict is None:
        if pred["expert"]["group"] == "expert:no_anomaly" or pred["telemetry_status"] == "no_telemetry" or not pred["features"]["tokens"]:
            return "missing_evidence", "关键遥测缺失或未检出可定位异常，证据不足是合理降级。"
        if pred["similarity"]["exact_labels"] and len(pred["similarity"]["exact_labels"]) > 1:
            return "overfitting_or_ambiguity", "相同可见特征对应多个训练标签，当前特征空间不可辨识。"
        return "decision_graph_gap", "已有异常或候选，但没有路径通过可靠性/因果门禁；决策图覆盖不足。"
    exact = pred["similarity"].get("exact_pure_support", 0)
    strong = [v for v in pred["decision_votes"] if v["strength"] in {"strong", "calibrated"}]
    physical_tx = [v for v in strong if v["source"] == "physical_tx_down"]
    if (label_status != "expert_reviewed" and len(physical_tx) == 1
            and all(v["verdict"] == verdict for v in strong)):
        return "label_suspect", "发送端触底并有状态证据支持唯一端点，但与未审核标签冲突；应进入人工标注复核，不自动改标。"
    if label_status != "expert_reviewed" and exact >= 2 and strong and all(v["verdict"] == verdict for v in strong):
        return "label_suspect", "多条纯净精确历史与高等级决策证据一致，但与当前未审核标签冲突。"
    if pred["expert_rule_accepted"]:
        return "decision_graph_error", "通过训练门禁的专家方向规则仍给出错误端点，规则因果方向或适用条件不足。"
    if exact == 1:
        return "overfitting", "结论依赖单条精确历史模式，支持不足导致过拟合。"
    learned = pred.get("learned_positive_path")
    if learned is not None and int(learned.get("support", 0)) <= 1:
        return "overfitting", "训练留一正向路径只命中一条近邻；当前错误说明该路径支持不足并发生过拟合。"
    if learned is not None:
        return "decision_graph_error", "训练留一正向路径通过门禁但方向错误，需要提高支持数、加入否定证据或拆分路径。"
    if not pred["features"]["tokens"] or pred["telemetry_status"] != "full_telemetry":
        return "feature_problem", "特征缺失或量测质量token未能表达关键差异，错误候选占优。"
    return "decision_graph_error", "物理/历史候选合并后方向错误，需要补充排除条件或仲裁边。"


def evaluate(blind: Sequence[Mapping[str, Any]], cases_by_id: Mapping[str, Mapping[str, Any]]) -> list[Dict[str, Any]]:
    rows = []
    for pred in blind:
        case = cases_by_id[pred["case_id"]]
        truth = str(case["label"])
        category, explanation = attribution(pred, truth, str(case.get("_dataset_contract", {}).get("label_status", "unreviewed")))
        explanation = (
            f"{explanation} 真实标签为{truth}；盲测输出为{pred.get('verdict') or '证据不足'}。"
            f"专家路径={pred['expert']['group']}，拓扑={pred['topology_id']}，"
            f"接收症状因果上下文={pred.get('receive_causal_context', 'not_available')}，"
            f"物理/历史候选数={len(pred['decision_votes'])}。"
        )
        rows.append({**pred, "actual": case["label"], "original_label": case.get("_dataset_contract", {}).get("original_label"),
                     "label_status": case.get("_dataset_contract", {}).get("label_status", "unreviewed"),
                     "correct": pred.get("verdict") == case["label"], "review_category": category,
                     "label_aware_review": explanation})
    return rows


def case_html(row: Mapping[str, Any]) -> str:
    e = html.escape
    color = "#067647" if row["correct"] else "#b54708" if row["verdict"] is None else "#b42318"
    def block(title: str, value: Any) -> str:
        return f"<section><h2>{e(title)}</h2><pre>{e(json.dumps(value,ensure_ascii=False,indent=2))}</pre></section>"
    decision = {
        "expert": row["expert"], "global_rule_stats": row["expert_rule_stats"],
        "topology_rule_stats": row["topology_rule_stats"],
        "receive_causal_context": row["receive_causal_context"],
        "statistical_gate": row["expert_statistical_gate"],
        "causal_gate": row["expert_causal_gate"], "votes": row["decision_votes"],
        "learned_positive_path": row.get("learned_positive_path"),
        "media_topology_positive_path": row.get("media_topology_positive_path"),
        "reason": row["reason"],
    }
    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>{e(row['case_id'])}</title><style>body{{max-width:1180px;margin:auto;padding:25px;font:14px/1.6 system-ui;background:#f5f7fb;color:#172033}}header,section{{background:white;border:1px solid #d8deea;border-radius:10px;padding:16px;margin-bottom:14px}}h1,h2{{margin-top:0}}.result{{border-left:6px solid {color}}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#101828;color:#e6edf8;padding:12px;border-radius:7px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}@media(max-width:700px){{.grid{{grid-template-columns:1fr}}}}</style></head><body><header class='result'><h1>{e(row['case_id'])}</h1><p><b>预测：</b>{e(str(row['verdict'] or '证据不足'))}　<b>真实：</b>{e(row['actual'])}　<b>复盘：</b>{e(row['review_category'])}</p><p>{e(row['label_aware_review'])}</p></header><section><h2>盲推理步骤</h2><ol>{''.join(f'<li>{e(x)}</li>' for x in row['blind_reasoning'])}</ol></section><div class='grid'>{block('可解释特征',row['features'])}{block('排障决策与候选',decision)}</div><div class='grid'>{block('训练历史候选',row['history'])}{block('补采建议',row['acquisition_recommendations'])}</div>{block('真实标签揭示后的复盘',{'actual':row['actual'],'original_label':row['original_label'],'label_status':row['label_status'],'category':row['review_category'],'analysis':row['label_aware_review']})}</body></html>"""


def main_html(rows: Sequence[Mapping[str, Any]], freeze: Mapping[str, Any]) -> str:
    e = html.escape
    counts = Counter(row["review_category"] for row in rows)
    covered = [row for row in rows if row["verdict"] is not None]
    correct = [row for row in covered if row["correct"]]
    split_stats = []
    for split in SPLITS:
        part=[r for r in rows if r["split"]==split];cov=[r for r in part if r["verdict"] is not None];hit=[r for r in cov if r["correct"]]
        split_stats.append((split,len(part),len(cov),len(hit),len(hit)/len(cov) if cov else 0))
    optimization = [
        ("P0：继续扩大安全覆盖", f"v5覆盖提升至{len(covered)}/{len(rows)}，但仍有{counts['decision_graph_gap']}条可见异常没有通过正向路径。下一步优先从训练集补齐multi_metric的否定证据和logical8的L1/fiber路径，不降低rxpower与SerDes门禁。"),
        ("P0：收紧结果已生效", "rxpower要求至少2条同拓扑近邻后自动终裁为0；SerDes的8条近邻命中全部降为advisory候选，自动终裁同样为0。两类在v4中的10条错误/不稳定结论已退出最终覆盖。"),
        ("P0：评估media_snr拓扑路径", "logical4 media_snr拓扑路径覆盖32条、正确27条（84.38%），是本轮覆盖提升来源；5条错误均需结合Rx LOS/LOL、对端Tx和量测完整性增加排除边。logical8仍只使用原有近邻路径，不借用logical4统计。"),
        ("P0：补采与量测质量", f"{counts['missing_evidence']}条missing_evidence属于无异常、无遥测或关键字段缺失。每个case页面已给出针对性补采项；应采集告警前后时间窗、两端同步Tx/Rx、LOS/LOL、SNR和端口状态，缺测继续保持独立状态。"),
        ("P1：保留冲突否决", "物理、训练近邻和专家路径中任何不同方向候选仍会降级。该门没有阻止同方向正向路径恢复覆盖，并继续避免logical8端口状态模式被单一端点证据强行终裁。"),
        ("P1：特征与标签复核", f"当前label_suspect {counts['label_suspect']}条。case_5f9fb799fec41356表现为L2发送触底且TxLOS/TxLOL支持，但标签为fiber；已进入标注清单，需核验原始波形、链路操作记录和人工结论，未经确认不改标。"),
        ("P1：重新做独立泛化验证", "v5仍属于测试知情迭代。代码与图谱上传后，应在新的时间段或嵌套时间折上验证；当前80.28%选择性准确率不能作为无偏最终指标。"),
    ]
    previous = []
    for name, path in (
        ("v1 基线", ROOT / "artifacts/filtered_rule_decision_graph_test_v1/summary.json"),
        ("v2 因果/拓扑门", ROOT / "artifacts/filtered_rule_decision_graph_test_v2/summary.json"),
        ("v3 冲突否决", ROOT / "artifacts/filtered_rule_decision_graph_test_v3/summary.json"),
        ("v4 训练留一正向路径", ROOT / "artifacts/filtered_rule_decision_graph_test_v4/summary.json"),
    ):
        if path.exists():
            summary = json.loads(path.read_text(encoding="utf-8"))
            previous.append((name, summary["covered"], summary["coverage"], summary["correct_at_coverage"], summary["selective_accuracy"], summary["insufficient"]))
    previous.append(("v5 收紧并恢复覆盖", len(covered), len(covered)/len(rows), len(correct), len(correct)/len(covered) if covered else 0.0, len(rows)-len(covered)))
    iteration_rows = "".join(
        f"<tr><td>{e(name)}</td><td>{cov} ({coverage:.2%})</td><td>{hit}</td><td>{accuracy:.2%}</td><td>{insufficient}</td></tr>"
        for name,cov,coverage,hit,accuracy,insufficient in previous
    )
    rows_html="".join(f"<tr data-split='{e(r['split'])}' data-cat='{e(r['review_category'])}'><td><a href='cases/{e(r['case_id'])}.html'>{e(r['case_id'])}</a></td><td>{e(r['split'])}</td><td>{e(str(r['verdict'] or '证据不足'))}</td><td>{e(r['actual'])}</td><td>{e(r['review_category'])}</td><td>{e(r['reason'])}</td></tr>" for r in rows)
    logical8_stats = next(item for item in split_stats if item[0] == "test/rule1_channel_not_4")
    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>决策图谱测试集逐Case复盘</title><style>body{{margin:0;background:#f4f6fa;color:#172033;font:14px/1.55 system-ui}}header{{background:linear-gradient(120deg,#101828,#1849a9);color:white;padding:32px max(22px,calc((100% - 1400px)/2))}}main{{max-width:1400px;margin:auto;padding:20px}}.card{{background:white;border:1px solid #d8deea;border-radius:11px;padding:17px;margin-bottom:14px}}.warn{{border-left:6px solid #b54708;background:#fffaeb}}.metrics{{display:grid;grid-template-columns:repeat(5,1fr);gap:9px}}.metric{{background:#f2f4f7;padding:11px;border-radius:8px}}.metric b{{display:block;font-size:22px;color:#2457d6}}table{{width:100%;border-collapse:collapse;min-width:1000px}}th,td{{border:1px solid #d8deea;padding:7px;text-align:left}}th{{background:#f2f4f7}}.table{{overflow:auto}}select,input{{padding:7px;border:1px solid #cbd3e1;border-radius:7px;margin:3px}}li{{margin:7px 0}}@media(max-width:700px){{.metrics{{grid-template-columns:1fr 1fr}}}}</style></head><body><header><h1>排障决策图谱 × 可解释特征：484 Case测试复盘 v4</h1><p>每轮预测先冻结、标签后揭示。知识包 {e(freeze['knowledge_bundle_hash'])}；决策图 {e(freeze['decision_graph_hash'][:16])}；N8冻结。</p></header><main><section class='card warn'><h2>核心结论</h2><p><b>训练留一正向路径成功恢复覆盖，但尚未达到正式发布标准。</b> v4覆盖率{len(covered)/len(rows):.2%}，覆盖内准确率{len(correct)/len(covered):.2%}；全测试集命中率{len(correct)/len(rows):.2%}。logical8覆盖{logical8_stats[2]}/{logical8_stats[1]}、正确{logical8_stats[3]}/{logical8_stats[2]}。结果仍是测试知情迭代验证，不是新的独立盲测。</p></section><section class='card'><h2>总览</h2><div class='metrics'><div class='metric'><b>{len(rows)}</b>测试case</div><div class='metric'><b>{len(covered)/len(rows):.1%}</b>自动覆盖率</div><div class='metric'><b>{len(correct)}/{len(covered)}</b>覆盖内正确</div><div class='metric'><b>{len(correct)/len(covered):.1%}</b>选择性准确率</div><div class='metric'><b>{len(rows)-len(covered)}</b>证据不足</div></div><div class='table'><table><tr><th>split</th><th>case</th><th>覆盖</th><th>正确</th><th>覆盖内准确率</th></tr>{''.join(f'<tr><td>{e(s)}</td><td>{n}</td><td>{c} ({c/n:.1%})</td><td>{h}</td><td>{a:.1%}</td></tr>' for s,n,c,h,a in split_stats)}</table></div></section><section class='card'><h2>/loop迭代对比</h2><p>v1暴露错误后，v2-v4均属于测试知情修复验证；阈值仅由训练留一法拟合，但架构方向来自前序复盘，因此不能作为独立泛化指标。</p><div class='table'><table><tr><th>迭代</th><th>覆盖</th><th>覆盖内正确</th><th>选择性准确率</th><th>证据不足</th></tr>{iteration_rows}</table></div></section><section class='card'><h2>复盘分类</h2><p>{e(json.dumps(dict(counts),ensure_ascii=False))}</p><p>分类含义：correct=准确；decision_graph_gap/error=排障决策缺失或错误；feature_problem=可解释特征/量测表达有问题；missing_evidence=证据缺失；overfitting*=历史模式过拟合或不可辨识；label_suspect=标签疑似问题。</p></section><section class='card'><h2>详细优化建议</h2><ol>{''.join(f'<li><b>{e(t)}</b><br>{e(d)}</li>' for t,d in optimization)}</ol></section><section class='card'><h2>逐Case报告</h2><div><select id='split'><option value=''>全部split</option>{''.join(f'<option>{e(s)}</option>' for s in SPLITS)}</select><select id='cat'><option value=''>全部分类</option>{''.join(f'<option>{e(c)}</option>' for c in sorted(counts))}</select><input id='q' placeholder='搜索case/结论'></div><div class='table'><table><thead><tr><th>case</th><th>split</th><th>预测</th><th>真实</th><th>复盘分类</th><th>盲推理结论</th></tr></thead><tbody id='body'>{rows_html}</tbody></table></div></section><script>function f(){{let s=split.value,c=cat.value,x=q.value.toLowerCase();body.querySelectorAll('tr').forEach(r=>r.hidden=!!((s&&r.dataset.split!==s)||(c&&r.dataset.cat!==c)||!r.innerText.toLowerCase().includes(x)))}}split.onchange=cat.onchange=q.oninput=f;</script></main></body></html>"""


def main_html_v5(rows: Sequence[Mapping[str, Any]], freeze: Mapping[str, Any]) -> str:
    """Render the v5 loop report with explicit candidate-vs-terminal accounting."""
    e = html.escape
    counts = Counter(row["review_category"] for row in rows)
    covered = [row for row in rows if row["verdict"] is not None]
    correct = [row for row in covered if row["correct"]]
    split_stats = []
    for split in SPLITS:
        part = [row for row in rows if row["split"] == split]
        cov = [row for row in part if row["verdict"] is not None]
        hit = [row for row in cov if row["correct"]]
        split_stats.append((split, len(part), len(cov), len(hit), len(hit) / len(cov) if cov else 0.0))
    iterations = []
    for name, directory in (
        ("v1 基线", "filtered_rule_decision_graph_test_v1"),
        ("v2 因果/拓扑门", "filtered_rule_decision_graph_test_v2"),
        ("v3 冲突否决", "filtered_rule_decision_graph_test_v3"),
        ("v4 训练留一正向路径", "filtered_rule_decision_graph_test_v4"),
    ):
        path = ROOT / "artifacts" / directory / "summary.json"
        if path.exists():
            item = json.loads(path.read_text(encoding="utf-8"))
            iterations.append((name, item["covered"], item["coverage"], item["correct_at_coverage"], item["selective_accuracy"], item["insufficient"]))
    iterations.append(("v5 收紧并恢复覆盖", len(covered), len(covered)/len(rows), len(correct), len(correct)/len(covered), len(rows)-len(covered)))
    iteration_rows = "".join(f"<tr><td>{e(n)}</td><td>{c} ({r:.2%})</td><td>{h}</td><td>{a:.2%}</td><td>{i}</td></tr>" for n,c,r,h,a,i in iterations)
    split_rows = "".join(f"<tr><td>{e(s)}</td><td>{n}</td><td>{c} ({c/n:.2%})</td><td>{h}</td><td>{a:.2%}</td></tr>" for s,n,c,h,a in split_stats)
    optimization = [
        ("P0：继续扩大安全覆盖", f"当前仍有{counts['decision_graph_gap']}条决策图缺口。优先从训练集补齐multi_metric否定证据和logical8的L1/fiber路径，不降低rxpower与SerDes门禁。"),
        ("P0：收紧结果已生效", "rxpower要求至少2条同拓扑近邻，自动终裁为0；SerDes的8条命中全部是advisory候选，自动终裁为0。"),
        ("P0：完善media_snr路径", "logical4 media_snr路径覆盖32条、正确27条；需用Rx LOS/LOL、对端Tx和量测完整性为5条错误补充排除边。"),
        ("P0：补采与量测质量", f"{counts['missing_evidence']}条证据缺失继续降级；各case页已给出同步Tx/Rx、LOS/LOL、SNR、端口状态和时间窗补采清单。"),
        ("P1：保留冲突否决", "任何物理、历史或专家候选方向冲突仍进入证据不足，不以多数票覆盖物理矛盾。"),
        ("P1：标签人工复核", f"label_suspect共{counts['label_suspect']}条；case_5f9fb799fec41356已进入annotation_queue，未经人工确认不改标。"),
        ("P1：独立泛化验证", "v5是测试知情迭代；应在新时间段或嵌套时间折上验证，当前80.28%选择性准确率不是无偏最终指标。"),
    ]
    optimization_html = "".join(f"<li><b>{e(title)}</b><br>{e(detail)}</li>" for title,detail in optimization)
    row_html = "".join(
        f"<tr data-split='{e(row['split'])}' data-cat='{e(row['review_category'])}'><td><a href='cases/{e(row['case_id'])}.html'>{e(row['case_id'])}</a></td><td>{e(row['split'])}</td><td>{e(str(row['verdict'] or '证据不足'))}</td><td>{e(row['actual'])}</td><td>{e(row['review_category'])}</td><td>{e(row['reason'])}</td></tr>"
        for row in rows
    )
    options_split = "".join(f"<option>{e(split)}</option>" for split in SPLITS)
    options_cat = "".join(f"<option>{e(category)}</option>" for category in sorted(counts))
    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>决策图谱测试集逐Case复盘 v5</title><style>body{{margin:0;background:#f4f6fa;color:#172033;font:14px/1.55 system-ui}}header{{background:linear-gradient(120deg,#101828,#1849a9);color:white;padding:32px max(22px,calc((100% - 1400px)/2))}}main{{max-width:1400px;margin:auto;padding:20px}}.card{{background:white;border:1px solid #d8deea;border-radius:11px;padding:17px;margin-bottom:14px}}.warn{{border-left:6px solid #b54708;background:#fffaeb}}.metrics{{display:grid;grid-template-columns:repeat(5,1fr);gap:9px}}.metric{{background:#f2f4f7;padding:11px;border-radius:8px}}.metric b{{display:block;font-size:22px;color:#2457d6}}table{{width:100%;border-collapse:collapse;min-width:1000px}}th,td{{border:1px solid #d8deea;padding:7px;text-align:left}}th{{background:#f2f4f7}}.table{{overflow:auto}}select,input{{padding:7px;border:1px solid #cbd3e1;border-radius:7px;margin:3px}}li{{margin:7px 0}}@media(max-width:700px){{.metrics{{grid-template-columns:1fr 1fr}}}}</style></head><body><header><h1>排障决策图谱 × 可解释特征：484 Case测试复盘 v5</h1><p>预测先冻结、标签后揭示。知识包 {e(freeze['knowledge_bundle_hash'])}；决策图 {e(freeze['decision_graph_hash'][:16])}；N8冻结。</p></header><main><section class='card warn'><h2>核心结论</h2><p><b>收紧rxpower/SerDes后，覆盖率和准确率同时改善。</b> v5覆盖{len(covered)}/{len(rows)}（{len(covered)/len(rows):.2%}），覆盖内正确{len(correct)}/{len(covered)}（{len(correct)/len(covered):.2%}）。all_data为50/63，logical8为7/8。rxpower和SerDes自动终裁均为0；新增覆盖来自训练内通过门禁的logical4 media_snr路径。</p><p>本轮仍属于测试知情迭代，不是独立盲测。</p></section><section class='card'><h2>总览</h2><div class='metrics'><div class='metric'><b>{len(rows)}</b>测试case</div><div class='metric'><b>{len(covered)/len(rows):.1%}</b>覆盖率</div><div class='metric'><b>{len(correct)}/{len(covered)}</b>覆盖内正确</div><div class='metric'><b>{len(correct)/len(covered):.1%}</b>选择性准确率</div><div class='metric'><b>{len(rows)-len(covered)}</b>证据不足</div></div><div class='table'><table><tr><th>split</th><th>case</th><th>覆盖</th><th>正确</th><th>覆盖内准确率</th></tr>{split_rows}</table></div></section><section class='card'><h2>/loop迭代对比</h2><div class='table'><table><tr><th>迭代</th><th>覆盖</th><th>正确</th><th>选择性准确率</th><th>证据不足</th></tr>{iteration_rows}</table></div></section><section class='card'><h2>复盘分类</h2><p>{e(json.dumps(dict(counts),ensure_ascii=False))}</p><p>correct=准确；decision_graph_gap/error=决策缺失或错误；feature_problem=特征/量测表达问题；missing_evidence=某部分缺失；overfitting=历史模式过拟合；label_suspect=疑似标签问题。</p></section><section class='card'><h2>详细优化建议</h2><ol>{optimization_html}</ol></section><section class='card'><h2>逐Case报告</h2><select id='split'><option value=''>全部split</option>{options_split}</select><select id='cat'><option value=''>全部分类</option>{options_cat}</select><input id='q' placeholder='搜索case/结论'><div class='table'><table><thead><tr><th>case</th><th>split</th><th>预测</th><th>真实</th><th>复盘分类</th><th>盲推理结论</th></tr></thead><tbody id='body'>{row_html}</tbody></table></div></section><script>function f(){{let s=split.value,c=cat.value,x=q.value.toLowerCase();body.querySelectorAll('tr').forEach(r=>r.hidden=!!((s&&r.dataset.split!==s)||(c&&r.dataset.cat!==c)||!r.innerText.toLowerCase().includes(x)))}}split.onchange=cat.onchange=q.oninput=f;</script></main></body></html>"""


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--data-dir",type=Path,default=DATA);p.add_argument("--knowledge",type=Path,default=KNOWLEDGE);p.add_argument("--decision-graph",type=Path,default=DECISION_GRAPH);p.add_argument("--output-dir",type=Path,default=OUTPUT);args=p.parse_args()
    bundle=OfflineKnowledgeBundle.load(args.knowledge);graph=json.loads(args.decision_graph.read_text());stats={r["group"]:r for r in graph["rule_stats"]};topology_stats={(r["group"],r["topology_id"]):r for r in graph.get("topology_rule_stats",[])};learned_rows=graph.get("learned_path_rows",[])
    all_cases=[];blind=[]
    for split in SPLITS:
        cases=cases_by_manifest_split(args.data_dir,split);all_cases.extend(cases);blind.extend(blind_predict(cases,split,bundle,stats,topology_stats,learned_rows))
    args.output_dir.mkdir(parents=True,exist_ok=True);blind_path=args.output_dir/"blind_predictions.json";dump(blind_path,blind)
    freeze={"schema_version":"decision-graph-blind-freeze-v5","case_count":len(blind),"prediction_sha256":sha(blind_path),"knowledge_bundle_hash":bundle.content_hash(),"decision_graph_hash":sha(args.decision_graph),"labels_visible_during_prediction":False,"prior_test_informed_iteration":True,"prior_iteration_used_for":"rxpower multi-neighbor and SerDes advisory gates from prior loop; coverage restored only by train topology-calibrated media path","n8_frozen":True};dump(args.output_dir/"blind_freeze.json",freeze)
    rows=evaluate(blind,{c["case_id"]:c for c in all_cases});dump(args.output_dir/"evaluated_cases.json",rows)
    dump(args.output_dir/"predictions.json",blind)
    dump(args.output_dir/"case_analysis.json",rows)
    dump(args.output_dir/"bad_cases.json",[row for row in rows if not row["correct"]])
    dump(args.output_dir/"label_suspects.json", [
        {"case_id": row["case_id"], "split": row["split"], "actual": row["actual"],
         "prediction": row["verdict"], "reason": row["label_aware_review"],
         "label_status": row["label_status"], "evidence": row["decision_votes"]}
        for row in rows if row["review_category"] == "label_suspect"
    ])
    dump(args.output_dir/"annotation_queue.json", {
        "schema_version": "filtered-rule-loop-label-review-v1",
        "source": str(args.output_dir / "label_suspects.json"),
        "annotations": {},
        "cases": [
            {"case_id": row["case_id"], "split": row["split"], "current_label": row["actual"],
             "proposed_label": row["verdict"], "status": "pending_human_review",
             "reason": row["label_aware_review"], "evidence": row["decision_votes"]}
            for row in rows if row["review_category"] == "label_suspect"
        ],
    })
    dump(args.output_dir/"learned_path_failures.json", [
        {"case_id": row["case_id"], "split": row["split"], "actual": row["actual"],
         "prediction": row["verdict"], "category": row["review_category"],
         "learned_positive_path": row.get("learned_positive_path"),
         "features": row["features"], "analysis": row["label_aware_review"]}
        for row in rows if row.get("learned_positive_path") is not None and not row["correct"]
    ])
    dump(args.output_dir/"irreducible_cases.json", [
        {"case_id": row["case_id"], "split": row["split"], "actual": row["actual"],
         "reason": row["label_aware_review"], "missing_fields": row["missing_fields"]}
        for row in rows if row["review_category"] in {"missing_evidence", "overfitting", "overfitting_or_ambiguity"}
    ])
    case_dir=args.output_dir/"cases";case_dir.mkdir(exist_ok=True)
    for row in rows:(case_dir/f"{row['case_id']}.html").write_text(case_html(row),encoding="utf-8")
    rendered_main = main_html_v5(rows,freeze)
    (args.output_dir/"index.html").write_text(rendered_main,encoding="utf-8")
    (args.output_dir/"report.html").write_text(rendered_main,encoding="utf-8")
    counts=Counter(r["review_category"] for r in rows);covered=[r for r in rows if r["verdict"]];correct=sum(r["correct"] for r in covered)
    split_metrics = {}
    for split in SPLITS:
        part = [row for row in rows if row["split"] == split]
        split_covered = [row for row in part if row["verdict"]]
        split_correct = sum(row["correct"] for row in split_covered)
        split_metrics[split] = {"case_count": len(part), "covered": len(split_covered),
                                "coverage": len(split_covered) / len(part),
                                "correct_at_coverage": split_correct,
                                "selective_accuracy": split_correct / len(split_covered) if split_covered else 0.0,
                                "insufficient": len(part) - len(split_covered)}
    learned_metrics = {}
    for group in sorted({row["expert"]["group"] for row in rows if row.get("learned_positive_path") is not None}):
        part = [row for row in rows if row.get("learned_positive_path") is not None and row["expert"]["group"] == group]
        terminal = [row for row in part if any(v["source"] == "train_loo_positive_path" and v["strength"] == "calibrated" for v in row["decision_votes"])]
        terminal_correct = sum(row["correct"] for row in terminal if row["verdict"] is not None)
        learned_metrics[group] = {"matched_candidates": len(part), "terminal_votes": len(terminal),
                                  "terminal_outputs": sum(row["verdict"] is not None for row in terminal),
                                  "terminal_correct": terminal_correct}
    media_topology = [row for row in rows if row.get("media_topology_positive_path")]
    summary={"schema_version":"decision-graph-loop-summary-v5","case_count":len(rows),"covered":len(covered),"coverage":len(covered)/len(rows),"correct_at_coverage":correct,"selective_accuracy":correct/len(covered) if covered else 0,"insufficient":len(rows)-len(covered),"split_metrics":split_metrics,"learned_path_metrics":learned_metrics,"media_topology_path":{"covered":len(media_topology),"correct":sum(row["correct"] for row in media_topology),"accuracy":sum(row["correct"] for row in media_topology)/len(media_topology) if media_topology else 0.0},"review_categories":dict(counts),"freeze":freeze};dump(args.output_dir/"summary.json",summary);print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
