#!/usr/bin/env python3
"""Build a train-only expert/physical decision graph and interactive HTML review."""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca_framework.data import cases_by_manifest_split, split_manifest_hash  # noqa: E402
from rca_framework.decision_graph_policy import (  # noqa: E402
    LEARNED_PATH_CONFIGS, learned_path_match, receive_symptom_context,
)
from rca_framework.evidence_pack import build_packs  # noqa: E402
from rca_framework.expert import (  # noqa: E402
    ANOMALY_ORDER, DOC_VARIANT, EXPERT_THRESHOLDS, ExpertVariant,
    detect_side_anomalies, diagnose, diagnose_side, port_down, side_metric_values,
)
from rca_framework.knowledge import OfflineKnowledgeBundle  # noqa: E402
from rca_framework.types import SIDES, wilson_lower_bound  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/filtered_rule_temporal_2025_06_09_v1"
KNOWLEDGE = ROOT / "artifacts/filtered_rule_deterministic_knowledge_v2/knowledge_bundle.json"
OUTPUT = ROOT / "artifacts/filtered_rule_decision_graph_v4"

METRIC_PLAIN = {
    "rxpower": "接收光是否异常", "txpower": "发送光是否异常", "host_snr": "主机侧电信号质量",
    "media_snr": "介质侧信号质量", "serdes_snr": "SerDes数字侧质量",
}


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def node(node_id: str, kind: str, layer: int, title: str, professional: str, plain: str,
         *, source: str, status: str = "active", **attrs: Any) -> Dict[str, Any]:
    return {"id": node_id, "kind": kind, "layer": layer, "title": title,
            "professional": professional, "plain": plain, "source": source,
            "status": status, "attrs": attrs}


def edge(src: str, dst: str, kind: str, condition: str, *, source: str, **attrs: Any) -> Dict[str, Any]:
    return {"src": src, "dst": dst, "kind": kind, "condition": condition,
            "source": source, "attrs": attrs}


def skeleton() -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    nodes = [
        node("start", "Input", 0, "标准化EvidencePack", "统一端点、指标、lane宽度和缺失状态。", "把原始工单整理成统一格式。", source="data-contract"),
        node("quality", "QualityGate", 1, "量测质量门", "检查no/partial telemetry、host_snr无效、宽度冲突和采集哨兵。", "先确认数据够不够用。", source="measurement-contract"),
        node("port_gate", "Predicate", 2, "端口状态门", "仅当txpower与rxpower均无有效发光时标记该端Down；Down是高严重度上下文，不再直接终裁。", "判断整端口是否已经完全掉线。", source="expert+review", status="revised"),
        node("side_eval", "Fork", 3, "L1/L2对称评估", "对两端执行完全相同的谓词与模式，不绑定速率和lane数量。", "两端用同一把尺子检查。", source="architecture"),
    ]
    edges = [edge("start", "quality", "next", "always", source="architecture"),
             edge("quality", "port_gate", "pass", "telemetry evaluable", source="measurement-contract"),
             edge("quality", "insufficient", "degrade", "critical telemetry missing", source="measurement-contract"),
             edge("port_gate", "side_eval", "next", "continue collecting causal evidence", source="review")]

    metrics = ("rxpower", "txpower", "host_snr", "media_snr", "serdes_snr")
    for metric in metrics:
        for kind in ANOMALY_ORDER:
            nid = f"pred:{metric}:{kind}"
            nodes.append(node(nid, "MetricPredicate", 4, f"{metric} · {kind}",
                              f"按专家阈值表判断{metric}的{kind}，保留多谓词而非短路后丢弃其它状态。",
                              f"{METRIC_PLAIN[metric]}：{kind}。", source="expert-document",
                              threshold=EXPERT_THRESHOLDS[metric].get({"lane_down":"down","low_value":"low","high_value":"high","lane_diff":"diff"}[kind])))
            edges.append(edge("side_eval", nid, "evaluate", f"for each side: {metric}", source="expert-document"))

    patterns = [
        ("pattern:tx_down", "发送关闭模式", "txpower lane_down直接支持异常端发送侧根因。", "发送光已经掉了，优先查这一端。", "expert-document"),
        ("pattern:multi", "多指标组合模式", "serdes+media+rx组合仅形成候选；训练冲突较高，必须经过因果门禁。", "多个指标一起异常，但不能直接拍板。", "expert+review"),
        ("pattern:local_digital", "本地数字侧模式", "host_snr/serdes_snr异常支持同侧设备候选。", "设备内部数字信号差，先怀疑这一端。", "expert-document"),
        ("pattern:receive_symptom", "接收侧症状模式", "rxpower/media_snr异常是接收症状，需检查对端Tx与介质路径后再定向。", "收得差不等于接收端坏了，还要查发送端和光纤。", "physics+review"),
        ("pattern:paired_loss", "跨端同lane方向模式", "tx_ok_rx_down支持该方向介质/接收路径；tx_down支持发送端；双向同lane异常形成fiber候选。", "利用同号lane判断光在哪里断。", "data-contract+physics"),
    ]
    for pid, title, pro, plain, source in patterns:
        nodes.append(node(pid, "Pattern", 5, title, pro, plain, source=source))
    for kind in ANOMALY_ORDER:
        edges.append(edge(f"pred:txpower:{kind}", "pattern:tx_down" if kind == "lane_down" else "pattern:local_digital", "supports", kind, source="expert-document"))
        edges.append(edge(f"pred:host_snr:{kind}", "pattern:local_digital", "supports", kind, source="expert-document"))
        edges.append(edge(f"pred:serdes_snr:{kind}", "pattern:local_digital", "supports", kind, source="expert-document"))
        edges.append(edge(f"pred:rxpower:{kind}", "pattern:receive_symptom", "supports", kind, source="physics+review"))
        edges.append(edge(f"pred:media_snr:{kind}", "pattern:receive_symptom", "supports", kind, source="physics+review"))

    nodes += [
        node("causal_gate", "CausalGate", 6, "方向与因果门禁", "组合本端Tx、对端Rx、LOS/LOL、paired-lane和缺失证据，区分发送端、接收端与介质。", "把症状沿光传播方向追到可能原因。", source="physics+review"),
        node("merge", "Arbitration", 7, "双端候选仲裁", "比较两端候选、证据等级、冲突与缺失；同优先级相反定界只表示冲突，不自动等于fiber。", "把两端检查结果放在一起判断。", source="expert+review", status="revised"),
        node("out:L1", "Outcome", 8, "L1 / 本端", "证据链支持本端设备或发送/数字侧根因。", "问题更可能在本端。", source="label-contract"),
        node("out:L2", "Outcome", 8, "L2 / 对端", "证据链支持对端设备或发送/数字侧根因。", "问题更可能在对端。", source="label-contract"),
        node("out:fiber", "Outcome", 8, "fiber / 介质", "需要发送健康且跨端接收异常、双向介质一致性或已确认历史模式。", "两端设备发光正常，但中间链路有问题。", source="physics+review"),
        node("insufficient", "Fallback", 8, "证据不足", "无异常、关键字段缺失、候选冲突或低支持路径进入补采/人工，不默认L1。", "现在的数据还不能可靠判断。", source="review", status="revised"),
    ]
    for pid, *_ in patterns:
        edges.append(edge(pid, "causal_gate", "candidate", "pattern matched", source="architecture"))
    edges += [edge("causal_gate", "merge", "validated", "causal checks pass", source="physics+review"),
              edge("causal_gate", "insufficient", "degrade", "critical evidence absent/conflicting", source="review"),
              edge("merge", "out:L1", "decide", "L1 evidence dominates", source="architecture"),
              edge("merge", "out:L2", "decide", "L2 evidence dominates", source="architecture"),
              edge("merge", "out:fiber", "decide", "positive medium evidence", source="physics+review"),
              edge("merge", "insufficient", "degrade", "low confidence or unresolved tie", source="review")]
    return nodes, edges


def build(data_dir: Path, bundle: OfflineKnowledgeBundle) -> Dict[str, Any]:
    cases = cases_by_manifest_split(data_dir, "train")
    packs = build_packs(cases, source_dataset=data_dir.name)
    feature_by_id = {row.case_id: row for row in bundle.training_features}
    no_fallback = ExpertVariant(name="decision-graph-no-fallback", single_metric_direction=DOC_VARIANT.single_metric_direction, use_fallbacks=False)
    templates: Dict[tuple[Any, ...], Dict[str, Any]] = {}
    rule_stats: Dict[str, Counter[str]] = defaultdict(Counter)
    topology_rule_stats: Dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    receive_context_stats: Dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    learned_path_rows = []
    covered = correct = 0
    for case, pack in zip(cases, packs):
        anomalies = {side: detect_side_anomalies(side_metric_values(pack, side)) for side in SIDES}
        sides = {side: diagnose_side(pack, side, variant=no_fallback) for side in SIDES}
        decision = diagnose(pack, variant=no_fallback)
        feature = feature_by_id[pack.case_id]
        learned_path_rows.append({"case_id": pack.case_id, "label": case["label"],
                                  "topology_id": pack.topology_id, "group": decision.group,
                                  "verdict": decision.verdict, "tokens": list(feature.tokens)})
        physical = tuple(sorted(token for token in feature.tokens if token.startswith(("status:", "lane:"))))
        key = (
            tuple(sorted((side, tuple(sorted(values.items()))) for side, values in anomalies.items())),
            tuple((side, sides[side].rule if sides[side] else "none", sides[side].location if sides[side] else "none") for side in SIDES),
            decision.group, physical,
        )
        row = templates.setdefault(key, {"support": 0, "labels": Counter(), "case_ids": [],
                                         "expert_predictions": Counter(), "anomalies": anomalies,
                                         "side_rules": {side: sides[side].to_dict() if sides[side] else None for side in SIDES},
                                         "decision_group": decision.group, "physical_tokens": list(physical)})
        row["support"] += 1; row["labels"][case["label"]] += 1; row["case_ids"].append(pack.case_id)
        row["expert_predictions"][str(decision.verdict or "insufficient")] += 1
        rule_stats[decision.group]["support"] += 1
        rule_stats[decision.group][f"truth:{case['label']}"] += 1
        rule_stats[decision.group]["correct"] += int(decision.verdict == case["label"])
        topology_key = (decision.group, pack.topology_id)
        topology_rule_stats[topology_key]["support"] += 1
        topology_rule_stats[topology_key][f"truth:{case['label']}"] += 1
        topology_rule_stats[topology_key]["correct"] += int(decision.verdict == case["label"])
        symptom_side = decision.sides[0].side if decision.sides else None
        context = receive_symptom_context(
            decision.group, decision.verdict, symptom_side, feature.tokens, anomalies,
        )
        if context != "not_applicable":
            context_key = (decision.group, pack.topology_id, context)
            receive_context_stats[context_key]["support"] += 1
            receive_context_stats[context_key][f"truth:{case['label']}"] += 1
            receive_context_stats[context_key]["correct"] += int(decision.verdict == case["label"])
        if decision.verdict is not None:
            covered += 1; correct += int(decision.verdict == case["label"])

    path_rows = []
    for index, row in enumerate(sorted(templates.values(), key=lambda item: (-item["support"], item["case_ids"][0])), 1):
        labels = row.pop("labels"); predictions = row.pop("expert_predictions")
        support = row["support"]
        path_rows.append({**row, "path_id": f"train-path-{index:03d}", "labels": dict(labels),
                          "expert_predictions": dict(predictions), "pure": len(labels) == 1,
                          "majority_label": labels.most_common(1)[0][0],
                          "majority_rate": labels.most_common(1)[0][1] / support})
    stats = []
    for group, counts in sorted(rule_stats.items()):
        total, hits = counts["support"], counts["correct"]
        stats.append({"group": group, "support": total, "correct": hits,
                      "accuracy": hits / total, "wilson_lower_bound": wilson_lower_bound(hits, total),
                      "truth_distribution": {label: counts[f"truth:{label}"] for label in ("L1", "L2", "fiber")}})
    topology_stats = []
    for (group, topology_id), counts in sorted(topology_rule_stats.items()):
        total, hits = counts["support"], counts["correct"]
        topology_stats.append({"group": group, "topology_id": topology_id, "support": total,
                               "correct": hits, "accuracy": hits / total,
                               "wilson_lower_bound": wilson_lower_bound(hits, total),
                               "truth_distribution": {label: counts[f"truth:{label}"] for label in ("L1", "L2", "fiber")}})
    context_stats = []
    for (group, topology_id, context), counts in sorted(receive_context_stats.items()):
        total, hits = counts["support"], counts["correct"]
        context_stats.append({"group": group, "topology_id": topology_id, "context": context,
                              "support": total, "correct": hits, "accuracy": hits / total,
                              "wilson_lower_bound": wilson_lower_bound(hits, total),
                              "truth_distribution": {label: counts[f"truth:{label}"] for label in ("L1", "L2", "fiber")}})
    learned_calibrations = []
    for group, config in LEARNED_PATH_CONFIGS.items():
        eligible = [row for row in learned_path_rows if row["group"] == group]
        accepted = []
        for row in eligible:
            match = learned_path_match(
                tokens=row["tokens"], topology_id=row["topology_id"], group=group,
                verdict=row["verdict"], training_rows=learned_path_rows,
                exclude_case_id=row["case_id"],
            )
            if match is not None:
                accepted.append((row, match))
        correct_loo = sum(row["label"] == match["verdict"] for row, match in accepted)
        learned_calibrations.append({"group": group, "config": dict(config),
                                     "eligible_train_cases": len(eligible),
                                     "loo_covered": len(accepted), "loo_correct": correct_loo,
                                     "loo_selective_accuracy": correct_loo / len(accepted) if accepted else 0.0,
                                     "loo_case_ids": [row["case_id"] for row, _ in accepted]})
    nodes, edges = skeleton()
    nodes += [
        node("receive_tx_gate", "CausalGate", 6, "接收症状发送端校验门",
             "rxpower/media_snr只描述接收症状；只有对端TxLOS/TxLOL、发送触底或同方向tx_down才允许形成对端根因票。",
             "收得差以后，还必须看到另一端确实发不出来，才能怪另一端。", source="physics+train-audit", status="revised"),
        node("topology_calibration", "CausalGate", 6, "拓扑分层可靠性门",
             "规则支持度按400G-200G logical4与400G-400G logical8分别计算，避免总体统计掩盖拓扑漂移。",
             "四路和八路的数据分别算可靠性，不能混在一起给规则背书。", source="train-audit", status="revised"),
        node("learned_positive_path", "CausalGate", 7, "训练留一正向路径门",
             "multi_metric、SerDes与单向接收症状只有在同拓扑、同规则、同方向近邻满足相似度、支持数和纯度时形成统计终裁候选。",
             "只有和同类历史故障足够像、且历史结论足够一致时才恢复自动判断。", source="train-loo", status="new"),
        node("media_topology_path", "CausalGate", 7, "Media SNR拓扑校准路径",
             "仅当当前拓扑上的media_snr规则支持数和Wilson下界通过门禁时，允许形成统计方向票；不外推到其他拓扑。",
             "某种接线中这条经验足够稳定时才启用，并且不能借给另一种接线。", source="train-topology-calibration", status="new"),
    ]
    edges += [
        edge("pattern:receive_symptom", "receive_tx_gate", "validate", "check opposite Tx and paired direction", source="physics+train-audit"),
        edge("receive_tx_gate", "merge", "validated", "positive opposite Tx fault", source="physics+train-audit"),
        edge("receive_tx_gate", "insufficient", "degrade", "healthy/unobserved Tx or missing direction", source="physics+train-audit"),
        edge("causal_gate", "topology_calibration", "calibrate", "evaluate within topology", source="train-audit"),
        edge("topology_calibration", "merge", "validated", "topology-specific reliability passes", source="train-audit"),
        edge("topology_calibration", "insufficient", "degrade", "topology support or lower bound fails", source="train-audit"),
        edge("topology_calibration", "learned_positive_path", "retrieve", "same topology, rule and direction", source="train-loo"),
        edge("learned_positive_path", "merge", "validated", "similarity/support/purity pass", source="train-loo"),
        edge("learned_positive_path", "insufficient", "degrade", "no calibrated positive path", source="train-loo"),
        edge("topology_calibration", "media_topology_path", "validate", "media_snr support>=8 and Wilson>=0.50", source="train-topology-calibration"),
        edge("media_topology_path", "merge", "validated", "topology-specific media path passes", source="train-topology-calibration"),
    ]
    return {
        "schema_version": "filtered-rule-decision-graph-v4", "source_document": "/Users/ziangchen/Downloads/专家模型.md",
        "train_manifest_hash": split_manifest_hash(data_dir), "knowledge_bundle_hash": bundle.content_hash(),
        "build_boundary": "expert rules + physical/data contracts + train-only path support; no test cases or labels",
        "n8_frozen": True, "nodes": nodes, "edges": edges, "train_case_count": len(cases),
        "expert_no_fallback_coverage": covered / len(cases), "expert_no_fallback_correct": correct,
        "expert_no_fallback_selective_accuracy": correct / covered if covered else 0.0,
        "path_count": len(path_rows), "singleton_path_count": sum(row["support"] == 1 for row in path_rows),
        "mixed_label_path_count": sum(not row["pure"] for row in path_rows),
        "rule_stats": stats, "topology_rule_stats": topology_stats,
        "receive_context_stats": context_stats, "learned_path_rows": learned_path_rows,
        "learned_path_calibrations": learned_calibrations, "train_paths": path_rows,
        "design_conclusions": [
            "无异常、端口双Down和未解决冲突统一进入证据不足，不再默认L1。",
            "专家规则是图谱骨架；训练标签只附着为路径支持度、分布和Wilson下界，不进入物理谓词。",
            "接收侧rxpower/media_snr异常先进入症状节点，必须经过Tx与paired-lane因果门禁才能指向端点或fiber。",
            "同优先级相反定界只表示候选冲突；fiber需要正向介质证据。",
            "低支持或混合标签训练路径保留审计，但不进入自动终裁。",
            "rxpower/media_snr总体训练准确率不能替代因果证据；训练集中这些路径没有opposite_tx_fault样本，因此不允许直接终裁端点。",
            "统计可靠性按topology_id分层；某一拓扑未通过Wilson门禁时，不借用另一拓扑的总体支持。",
            "multi_metric、SerDes和单向传播使用训练留一标定的正向近邻路径；相似度、支持或纯度不够时仍降级。",
            "rxpower至少需要两条近邻；SerDes近邻只保留为候选。覆盖恢复由同拓扑media_snr统计路径承担。",
        ],
    }


def render(graph: Mapping[str, Any]) -> str:
    e = html.escape
    payload = json.dumps({"nodes": graph["nodes"], "edges": graph["edges"], "paths": graph["train_paths"], "rules": graph["rule_stats"], "topology_rules": graph.get("topology_rule_stats", []), "receive_contexts": graph.get("receive_context_stats", []), "learned_calibrations": graph.get("learned_path_calibrations", [])}, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    conclusions = "".join(f"<li>{e(x)}</li>" for x in graph["design_conclusions"])
    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>训练增强排障决策图谱</title><style>
:root{{--bg:#f4f6fa;--panel:#fff;--ink:#172033;--muted:#667085;--line:#d8deea;--blue:#2457d6;--green:#067647;--amber:#b54708;--red:#b42318;--purple:#6941c6}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.6 system-ui,-apple-system,sans-serif}}header{{background:linear-gradient(120deg,#101828,#1849a9);color:#fff;padding:34px max(22px,calc((100% - 1450px)/2))}}header h1{{margin:0;font-size:30px}}header p{{max-width:1050px;color:#dce6ff}}main{{max-width:1450px;margin:auto;padding:22px}}.card{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:18px;margin-bottom:16px}}.metrics{{display:grid;grid-template-columns:repeat(5,1fr);gap:9px}}.metric{{background:#f7f9fc;padding:11px;border-radius:8px}}.metric b{{display:block;font-size:22px;color:var(--blue)}}.controls{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}}input,select{{border:1px solid #cbd3e1;border-radius:7px;padding:7px;font:inherit}}.graph{{display:grid;grid-template-columns:minmax(0,3fr) minmax(280px,1fr);gap:12px}}svg{{width:100%;min-height:760px;border:1px solid var(--line);background:#fbfcff;border-radius:9px}}.detail{{background:#f7f9fc;border-radius:9px;padding:12px;overflow-wrap:anywhere}}.detail pre{{white-space:pre-wrap;font-size:12px}}.edge{{stroke:#98a2b3;stroke-width:1.15;fill:none;opacity:.62}}.edge.active{{stroke:var(--blue);stroke-width:2.5;opacity:1}}.node{{cursor:pointer}}.node rect{{stroke:#fff;stroke-width:2}}.node text{{font-size:10.5px;pointer-events:none}}table{{width:100%;border-collapse:collapse;min-width:900px}}th,td{{border:1px solid var(--line);padding:7px;text-align:left;vertical-align:top}}th{{background:#f2f4f7}}.table{{overflow:auto}}.risk{{background:#fff5f4}}.ok{{background:#f6fef9}}.pill{{display:inline-block;padding:2px 7px;border-radius:999px;background:#eef2f8;margin:2px}}code{{font-size:12px}}.path{{display:grid;grid-template-columns:110px 70px 150px 1fr 160px;gap:8px;padding:8px;border-bottom:1px solid var(--line)}}.mixed{{background:#fff5f4}}@media(max-width:900px){{.metrics{{grid-template-columns:1fr 1fr}}.graph{{grid-template-columns:1fr}}.path{{grid-template-columns:1fr}}}}</style></head><body><header><h1>训练增强排障决策图谱 v4</h1><p>专家文档提供骨架，物理与量测契约提供因果门禁。rxpower取消单近邻终裁，SerDes降为候选；logical4 media_snr使用训练拓扑门恢复覆盖。测试case与测试label未参与构建。</p></header><main>
<section class='card'><h2>整体结果</h2><div class='metrics'><div class='metric'><b>{len(graph['nodes'])}</b>决策节点</div><div class='metric'><b>{len(graph['edges'])}</b>语义边</div><div class='metric'><b>{graph['path_count']}</b>训练路径模板</div><div class='metric'><b>{graph['expert_no_fallback_coverage']:.1%}</b>专家骨架覆盖</div><div class='metric'><b>{graph['expert_no_fallback_selective_accuracy']:.1%}</b>覆盖内准确率</div></div><p><span class='pill'>train only</span><span class='pill'>N8 frozen</span><span class='pill'>no test tuning</span></p><ul>{conclusions}</ul></section>
<section class='card'><h2>可交互决策图</h2><div class='controls'><select id='sourceFilter'><option value=''>全部知识来源</option></select><select id='kindFilter'><option value=''>全部节点类型</option></select><input id='search' placeholder='搜索节点'></div><div class='graph'><svg id='svg' viewBox='0 0 1200 760'></svg><div id='detail' class='detail'><b>点击节点</b><p>查看专业解释、通俗解释、来源与阈值。</p></div></div></section>
<section class='card'><h2>专家规则在训练集上的路径可靠性</h2><p>这些数字是统计层，不是物理规则。低支持和低Wilson下界路径只能用于候选或补采。</p><div id='rules' class='table'></div></section>
<section class='card'><h2>训练case扩展出的路径模板</h2><div class='controls'><select id='pathFilter'><option value=''>全部路径</option><option value='mixed'>仅混合标签</option><option value='singleton'>仅单例</option></select><input id='pathSearch' placeholder='搜索case/规则/token'></div><div id='paths'></div></section>
<script id='data' type='application/json'>{payload}</script><script>
const D=JSON.parse(document.getElementById('data').textContent),C={{Input:'#2457d6',QualityGate:'#475467',Predicate:'#667085',Fork:'#2457d6',MetricPredicate:'#12b76a',Pattern:'#f79009',CausalGate:'#6941c6',Arbitration:'#7f56d9',Outcome:'#f04438',Fallback:'#b42318'}};const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const sources=[...new Set(D.nodes.map(n=>n.source))].sort(),kinds=[...new Set(D.nodes.map(n=>n.kind))].sort();sourceFilter.innerHTML+=sources.map(x=>`<option>${{esc(x)}}</option>`).join('');kindFilter.innerHTML+=kinds.map(x=>`<option>${{esc(x)}}</option>`).join('');
function draw(){{let sf=sourceFilter.value,kf=kindFilter.value,q=search.value.toLowerCase(),visible=D.nodes.filter(n=>(!sf||n.source===sf)&&(!kf||n.kind===kf)&&JSON.stringify(n).toLowerCase().includes(q)),ids=new Set(visible.map(n=>n.id)),layers={{}};visible.forEach(n=>(layers[n.layer]??=[]).push(n));Object.values(layers).forEach(a=>a.sort((x,y)=>x.id.localeCompare(y.id)));let pos={{}};Object.entries(layers).forEach(([l,a])=>a.forEach((n,i)=>pos[n.id]={{x:55+Number(l)*135,y:45+(665*(i+1)/(a.length+1))}}));svg.innerHTML=D.edges.filter(x=>ids.has(x.src)&&ids.has(x.dst)).map(x=>`<path class=edge data-src="${{x.src}}" data-dst="${{x.dst}}" d="M ${{pos[x.src].x+45}} ${{pos[x.src].y}} C ${{(pos[x.src].x+pos[x.dst].x)/2}} ${{pos[x.src].y}}, ${{(pos[x.src].x+pos[x.dst].x)/2}} ${{pos[x.dst].y}}, ${{pos[x.dst].x-45}} ${{pos[x.dst].y}}"><title>${{esc(x.kind+' · '+x.condition)}}</title></path>`).join('')+visible.map(n=>`<g class=node data-id="${{n.id}}" transform="translate(${{pos[n.id].x}},${{pos[n.id].y}})"><rect x=-46 y=-16 width=92 height=32 rx=6 fill="${{C[n.kind]||'#667085'}}"></rect><text fill=white text-anchor=middle y=4>${{esc(n.title.length>13?n.title.slice(0,12)+'…':n.title)}}</text></g>`).join('');svg.querySelectorAll('.node').forEach(el=>el.onclick=()=>{{let n=D.nodes.find(x=>x.id===el.dataset.id);svg.querySelectorAll('.edge').forEach(x=>x.classList.toggle('active',x.dataset.src===n.id||x.dataset.dst===n.id));detail.innerHTML=`<b>${{esc(n.title)}}</b><p><span class=pill>${{esc(n.kind)}}</span><span class=pill>${{esc(n.source)}}</span><span class=pill>${{esc(n.status)}}</span></p><h3>专业解释</h3><p>${{esc(n.professional)}}</p><h3>通俗解释</h3><p>${{esc(n.plain)}}</p><pre>${{esc(JSON.stringify(n.attrs,null,2))}}</pre>`}})}}sourceFilter.onchange=kindFilter.onchange=search.oninput=draw;draw();
rules.innerHTML=`<table><thead><tr><th>规则组</th><th>支持</th><th>正确</th><th>准确率</th><th>Wilson下界</th><th>训练标签分布</th></tr></thead><tbody>${{D.rules.map(r=>`<tr class="${{r.wilson_lower_bound<.5?'risk':'ok'}}"><td><code>${{esc(r.group)}}</code></td><td>${{r.support}}</td><td>${{r.correct}}</td><td>${{(100*r.accuracy).toFixed(1)}}%</td><td>${{(100*r.wilson_lower_bound).toFixed(1)}}%</td><td>${{esc(JSON.stringify(r.truth_distribution))}}</td></tr>`).join('')}}</tbody></table>`;
function drawPaths(){{let f=pathFilter.value,q=pathSearch.value.toLowerCase(),rows=D.paths.filter(p=>(!f||(f==='mixed'?!p.pure:p.support===1))&&JSON.stringify(p).toLowerCase().includes(q));paths.innerHTML=rows.map(p=>`<details class="${{p.pure?'path':'path mixed'}}"><summary><b>${{p.path_id}}</b> · support ${{p.support}} · labels ${{esc(JSON.stringify(p.labels))}} · ${{esc(p.decision_group)}}</summary><p><b>side rules</b> <code>${{esc(JSON.stringify(p.side_rules))}}</code></p><p><b>physical tokens</b> ${{esc(p.physical_tokens.join(' · ')||'none')}}</p><p><b>cases</b> ${{esc(p.case_ids.join(', '))}}</p></details>`).join('')}}pathFilter.onchange=drawPaths;pathSearch.oninput=drawPaths;drawPaths();
</script></main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--knowledge", type=Path, default=KNOWLEDGE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    graph = build(args.data_dir, OfflineKnowledgeBundle.load(args.knowledge))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dump(args.output_dir / "decision_graph.json", graph)
    (args.output_dir / "decision_graph.html").write_text(render(graph), encoding="utf-8")
    (args.output_dir / "README.md").write_text(
        "# Filtered-rule train-only decision graph v4\n\n"
        "专家规则是骨架，物理/量测契约是因果门禁，训练case只提供路径支持与冲突统计。\n\n"
        f"- nodes/edges: {len(graph['nodes'])}/{len(graph['edges'])}\n- train paths: {graph['path_count']}\n"
        f"- no-fallback coverage: {graph['expert_no_fallback_coverage']:.2%}\n",
        encoding="utf-8",
    )
    print(json.dumps({key: graph[key] for key in ("train_case_count", "expert_no_fallback_coverage", "expert_no_fallback_correct", "expert_no_fallback_selective_accuracy", "path_count", "singleton_path_count", "mixed_label_path_count")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
