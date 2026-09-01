#!/usr/bin/env python3
"""Aggregate frozen monthly test errors into a compact raw-case annotation tool.

This is a post-label audit artifact.  It never changes labels, thresholds,
knowledge, SOPs, or predictions.  Similar errors are grouped by observable
error/evidence structure and one representative is retained per group.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


SCHEMA = "aggregated-monthly-badcase-review-v1"
DEFAULT_INPUT = Path(
    "experiments/20260829_snapshot-l1l2-evolution/"
    "monthly_badcase_label_review/badcase_reviews.json"
)
DEFAULT_OUTPUT = Path(
    "experiments/20260829_snapshot-l1l2-evolution/"
    "aggregated_badcase_annotation_review"
)
DEFAULT_LLM_ANALYSIS = Path("artifacts/filtered_rule_full_test_analysis_v3/case_reviews.json")
LANE_METRICS = ("bias", "txpower", "rxpower", "transmission", "media_snr", "host_snr", "serdes_snr")
STATUS_FIELDS = ("RxLOS", "RxLOL", "TxLOS", "TxLOL")
SCALAR_FIELDS = (
    "Temperature", "Voltage", "Lane number", "vendor", "vendor_sn",
    "alarm_name", "alarm_time", "alarm_ip_interface", "link_location", "syslog",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def terminal_family(review: Mapping[str, Any]) -> str:
    paths = [
        path for path in review.get("causal_evidence", {}).get("paths", ())
        if path.get("terminal")
    ]
    families = {
        re.sub(r"_L[12]$", "", str(path.get("id", "unknown")))
        for path in paths
    }
    return "+".join(sorted(families)) or "none"


def evidence_family(review: Mapping[str, Any]) -> str:
    families = set()
    tokens = review.get("explainable_features", {}).get("detail_tokens", ())
    for token in tokens:
        parts = str(token).split(":")
        if str(token).startswith("status:") and len(parts) > 2:
            families.add(f"status_{parts[2]}")
        elif str(token).startswith("expert:") and len(parts) > 2:
            families.add(f"expert_{parts[2]}")
        elif str(token).startswith(("lane:", "lane_scope:")) and len(parts) > 2:
            families.add(f"lane_{parts[2]}")
        elif parts:
            families.add(parts[0])
    return "+".join(sorted(families)) or "empty"


def pattern_key(review: Mapping[str, Any]) -> tuple[str, ...]:
    analysis = review.get("post_label_analysis", {})
    sop = review.get("sop", {})
    return (
        str(analysis.get("primary_failure_layer", "unknown")),
        str(review.get("source_dataset", "unknown")),
        str(review.get("topology_id", "unknown")),
        f"{review.get('truth_label')}<-{review.get('model_prediction')}",
        str(review.get("selected_by", "unknown")),
        str(sop.get("group", "unknown")),
        terminal_family(review),
        evidence_family(review),
    )


def representative_rank(review: Mapping[str, Any]) -> tuple[Any, ...]:
    contract = review.get("raw_case", {}).get("_dataset_contract", {})
    reviewed_rank = 0 if contract.get("label_status") == "expert_reviewed" else 1
    path_count = len(review.get("causal_evidence", {}).get("paths", ()))
    similarity = float(review.get("evidence_graph", {}).get("top_similarity") or 0.0)
    return reviewed_rank, -path_count, -similarity, str(review.get("case_id"))


def metric_values(case: Mapping[str, Any], metric: str, endpoint: str) -> Dict[str, Any]:
    block = case.get(metric)
    if not isinstance(block, Mapping):
        return {}
    values = block.get(endpoint)
    return dict(values) if isinstance(values, Mapping) else {}


def direction_rows(case: Mapping[str, Any], sender: str, receiver: str) -> list[Dict[str, Any]]:
    direction = f"{sender}-{receiver}"
    transmission = case.get("transmission", {})
    direction_values = transmission.get(direction, {}) if isinstance(transmission, Mapping) else {}
    blocks = {
        "sender_txpower": metric_values(case, "txpower", sender),
        "receiver_rxpower": metric_values(case, "rxpower", receiver),
        "receiver_media_snr": metric_values(case, "media_snr", receiver),
        "receiver_serdes_snr": metric_values(case, "serdes_snr", receiver),
        "receiver_host_snr": metric_values(case, "host_snr", receiver),
        "transmission": dict(direction_values) if isinstance(direction_values, Mapping) else {},
    }
    lanes = sorted(
        {str(lane) for values in blocks.values() for lane in values},
        key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value),
    )
    return [{"lane": lane, **{name: values.get(lane) for name, values in blocks.items()}} for lane in lanes]


def raw_field_rows(case: Mapping[str, Any]) -> list[Dict[str, Any]]:
    rows = []
    for field in LANE_METRICS:
        block = case.get(field)
        if isinstance(block, Mapping):
            for endpoint, values in block.items():
                rows.append({"field": field, "endpoint": endpoint, "value": values})
        else:
            rows.append({"field": field, "endpoint": "—", "value": block})
    for field in STATUS_FIELDS + SCALAR_FIELDS:
        block = case.get(field)
        if isinstance(block, Mapping):
            for endpoint, value in block.items():
                rows.append({"field": field, "endpoint": endpoint, "value": value})
        else:
            rows.append({"field": field, "endpoint": "—", "value": block})
    return rows


def llm_analysis(record: Mapping[str, Any] | None, source: Path) -> Dict[str, Any]:
    if not record:
        return {
            "available": False,
            "source_artifact": str(source),
            "status": "该case没有可关联的历史LLM运行记录。",
            "steps": [],
        }
    return {
        "available": True,
        "source_artifact": str(source),
        "status": "历史冻结LLM分析记录；原样展示，不重新生成。",
        "remote_branch": record.get("remote_branch"),
        "remote_action": record.get("remote_action"),
        "llm_terminal_verdict": record.get("llm_terminal_verdict"),
        "llm_proposed_verdict": record.get("llm_proposed_verdict"),
        "llm_pre_reconcile_verdict": record.get("llm_pre_reconcile_verdict"),
        "llm_step_aggregate_verdict": record.get("llm_step_aggregate_verdict"),
        "llm_output_valid": record.get("llm_output_valid"),
        "confidence": record.get("confidence"),
        "confidence_lower_bound": record.get("confidence_lower_bound"),
        "confidence_breakdown": record.get("confidence_breakdown"),
        "steps": list(record.get("llm_steps", ())),
        "compliance_penalties": list(record.get("compliance_penalties", ())),
        "acquisition_recommendations": list(record.get("acquisition_recommendations", ())),
    }


def has_nonempty_llm_steps(record: Mapping[str, Any] | None) -> bool:
    """Only admit frozen LLM records containing at least one substantive step."""
    if not record:
        return False
    return any(
        isinstance(step, Mapping) and bool(str(step.get("statement", "")).strip())
        for step in (record.get("llm_steps") or record.get("steps") or ())
    )


def has_direct_llm_explanation(record: Mapping[str, Any] | None) -> bool:
    if not record:
        return False
    return any(
        isinstance(step, Mapping)
        and bool(str(step.get("statement", "")).strip())
        and str(step.get("source", "")).strip().lower() == "llm"
        for step in (record.get("llm_steps") or record.get("steps") or ())
    )


def has_explainable_features(review: Mapping[str, Any]) -> bool:
    features = review.get("explainable_features") or {}
    explanations = list(features.get("detail_explanations") or ()) + list(
        features.get("semantic_parent_explanations") or ()
    )
    return any(
        isinstance(item, Mapping)
        and bool(str(item.get("token", "")).strip())
        and bool(str(item.get("meaning", "")).strip())
        for item in explanations
    )


def build_cases(
    reviews: Sequence[Mapping[str, Any]],
    llm_by_case: Mapping[str, Mapping[str, Any]],
    llm_source: Path,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    grouped: Dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for review in reviews:
        if review.get("model_prediction") == review.get("truth_label"):
            continue
        record = llm_by_case.get(str(review["case_id"]))
        if not has_explainable_features(review) or not has_direct_llm_explanation(record):
            continue
        grouped[pattern_key(review)].append(review)

    ordered_groups = sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))
    selected = []
    patterns = []
    for index, (key, members) in enumerate(ordered_groups, start=1):
        pattern_id = f"PAT-{index:03d}"
        members = sorted(members, key=representative_rank)
        representative = members[0]
        case = representative["raw_case"]
        contract = case.get("_dataset_contract", {})
        patterns.append({
            "pattern_id": pattern_id,
            "case_count": len(members),
            "representative_case_id": representative["case_id"],
            "member_case_ids": sorted(str(row["case_id"]) for row in members),
            "failure_layer": key[0],
            "source_dataset": key[1],
            "topology_id": key[2],
            "confusion": key[3],
            "selected_by": key[4],
            "sop_group": key[5],
            "terminal_family": key[6],
            "evidence_family": key[7],
        })
        selected.append({
            "review_id": f"aggregated_{representative['case_id']}",
            "case_id": representative["case_id"],
            "pattern_id": pattern_id,
            "pattern_case_count": len(members),
            "pattern_member_case_ids": sorted(str(row["case_id"]) for row in members),
            "month": representative.get("month"),
            "source_dataset": representative.get("source_dataset"),
            "topology_id": representative.get("topology_id"),
            "current_label": representative.get("truth_label"),
            "model_prediction": representative.get("model_prediction"),
            "label_status": contract.get("label_status", "unreviewed"),
            "failure_layer": key[0],
            "selected_by": representative.get("selected_by"),
            "explainable_features": representative.get("explainable_features", {}),
            "measurement_quality": representative.get("measurement_quality", {}),
            "evidence_graph": {
                "retrieval_and_vote": representative.get("evidence_graph", {}),
                "causal_subgraph": representative.get("causal_evidence", {}),
                "neighbor_evidence_comparison": representative.get("neighbors", []),
            },
            "llm_analysis": llm_analysis(
                llm_by_case.get(str(representative["case_id"])), llm_source
            ),
            "direction_tables": {
                "L1_to_L2": direction_rows(case, "L1", "L2"),
                "L2_to_L1": direction_rows(case, "L2", "L1"),
            },
            "status_snapshot": {field: case.get(field) for field in STATUS_FIELDS},
            "case_metadata": {field: case.get(field) for field in SCALAR_FIELDS},
            "raw_field_rows": raw_field_rows(case),
            "raw_case": case,
        })
    return selected, patterns


def render_html(cases: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> str:
    payload = json.dumps({"cases": cases, "summary": summary}, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>聚合 Bad Case 审计标注工作台</title><style>
:root{{--bg:#f5f7fb;--panel:#fff;--ink:#172033;--muted:#667085;--line:#d9dfeb;--brand:#2855d9;--bad:#b42318;--ok:#067647;--warn:#b54708}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,-apple-system,"PingFang SC",sans-serif}}header{{position:sticky;top:0;z-index:5;background:#172554;color:white;padding:13px 20px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}}header h1{{font-size:18px;margin:0;flex:1}}button,input,select,textarea{{font:inherit}}button{{border:0;border-radius:7px;padding:8px 12px;cursor:pointer}}.primary{{background:#2563eb;color:white}}.ghost{{background:#dbeafe;color:#1e40af}}main{{display:grid;grid-template-columns:390px 1fr;min-height:calc(100vh - 62px)}}aside{{border-right:1px solid var(--line);background:white;position:sticky;top:62px;height:calc(100vh - 62px);overflow:auto}}.filters{{padding:12px;border-bottom:1px solid var(--line);display:grid;grid-template-columns:1fr 1fr;gap:7px}}.filters input{{grid-column:1/-1}}input,select,textarea{{border:1px solid #cbd5e1;border-radius:7px;padding:8px;min-width:0}}.item{{padding:11px 13px;border-bottom:1px solid #edf0f5;cursor:pointer}}.item:hover,.item.active{{background:#eff6ff}}.item.done{{border-left:5px solid #16a34a}}.pill{{display:inline-block;padding:2px 7px;margin:3px 4px 0 0;border-radius:999px;background:#e5eaf1;font-size:12px}}.danger{{background:#fee2e2;color:var(--bad)}}.good{{background:#dcfce7;color:var(--ok)}}.warn{{background:#fef3c7;color:var(--warn)}}.content{{padding:18px;min-width:0;max-width:1500px}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:15px;margin-bottom:13px}}.headline{{border-left:5px solid #dc2626}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:13px}}.scroll{{overflow:auto;max-height:520px}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;vertical-align:top;padding:8px;border-bottom:1px solid var(--line)}}th{{position:sticky;top:0;background:#f8fafc}}pre{{white-space:pre-wrap;word-break:break-word;background:#101828;color:#f2f4f7;padding:11px;border-radius:8px;max-height:500px;overflow:auto}}details{{margin:8px 0}}summary{{cursor:pointer;font-weight:650}}.form{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}label{{display:flex;flex-direction:column;gap:4px}}label.wide{{grid-column:1/-1}}textarea{{min-height:82px}}.members{{font-family:ui-monospace,SFMono-Regular,monospace;color:var(--muted);overflow-wrap:anywhere}}.feature{{padding:8px 0;border-bottom:1px dashed var(--line)}}.feature code{{overflow-wrap:anywhere}}.step{{border-left:3px solid #84adff;padding:8px 10px;margin:8px 0;background:#f8fbff}}.step.invalid{{border-left-color:#f59e0b;background:#fffbeb}}@media(max-width:920px){{main{{grid-template-columns:1fr}}aside{{position:relative;top:0;height:360px}}.grid,.form{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>聚合 Bad Case · 审计标注工作台</h1><span id='progress'></span><button class='primary' onclick='exportNotes()'>导出标注 JSON</button><button class='ghost' onclick='importer.click()'>导入</button><input id='importer' type='file' hidden accept='.json' onchange='importNotes(event)'></header><main><aside><div class='filters'><input id='search' placeholder='搜索 case / pattern / 特征 / 原始数据' oninput='renderList()'><select id='source' onchange='renderList()'><option value=''>全部来源</option><option>all_data</option><option>rule1_channel_not_4</option></select><select id='layer' onchange='renderList()'><option value=''>全部错误模式</option></select><select id='truth' onchange='renderList()'><option value=''>全部当前标签</option><option>L1</option><option>L2</option><option>fiber</option></select><select id='status' onchange='renderList()'><option value=''>全部标签状态</option><option>unreviewed</option><option>expert_reviewed</option></select><select id='state' onchange='renderList()'><option value=''>全部审核状态</option><option value='open'>未完成</option><option value='done'>已完成</option></select></div><div id='list'></div></aside><section class='content'><div id='detail'></div></section></main>
<script id='dataset' type='application/json'>{payload}</script><script>
const DATA=JSON.parse(document.getElementById('dataset').textContent),KEY='aggregated_badcase_annotations_v1';let current=DATA.cases[0]?.review_id||'',notes=JSON.parse(localStorage.getItem(KEY)||'{{}}');const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c])),pretty=x=>esc(JSON.stringify(x,null,2));
for(const x of [...new Set(DATA.cases.map(r=>r.failure_layer))])layer.insertAdjacentHTML('beforeend',`<option>${{esc(x)}}</option>`);
function filtered(){{let q=search.value.toLowerCase();return DATA.cases.filter(r=>(!source.value||r.source_dataset===source.value)&&(!layer.value||r.failure_layer===layer.value)&&(!truth.value||r.current_label===truth.value)&&(!status.value||r.label_status===status.value)&&(!state.value||(state.value==='done')===!!notes[r.review_id]?.completed)&&(!q||JSON.stringify(r).toLowerCase().includes(q)))}}
function renderList(){{let xs=filtered();list.innerHTML=xs.map(r=>`<div class='item ${{r.review_id===current?'active':''}} ${{notes[r.review_id]?.completed?'done':''}}' onclick="pick('${{r.review_id}}')"><b>${{esc(r.case_id)}}</b><br><span class='pill danger'>标签 ${{esc(r.current_label)}} / 预测 ${{esc(r.model_prediction)}}</span><span class='pill'>${{esc(r.pattern_id)}} · 同型 ${{r.pattern_case_count}}</span><br><small>${{esc(r.source_dataset)}} · ${{esc(r.failure_layer)}}</small></div>`).join('')||'<p style="padding:14px">无匹配 case</p>';if(!xs.some(r=>r.review_id===current))current=xs[0]?.review_id||'';updateProgress();renderDetail();enhanceDetail()}}
function laneTable(title,xs){{return `<h4>${{esc(title)}}</h4><div class='scroll'><table><tr><th>lane</th><th>发送Tx</th><th>接收Rx</th><th>接收media</th><th>接收SerDes</th><th>接收host</th><th>transmission</th></tr>${{xs.map(x=>`<tr><td>${{esc(x.lane)}}</td><td>${{esc(x.sender_txpower)}}</td><td>${{esc(x.receiver_rxpower)}}</td><td>${{esc(x.receiver_media_snr)}}</td><td>${{esc(x.receiver_serdes_snr)}}</td><td>${{esc(x.receiver_host_snr)}}</td><td>${{esc(x.transmission)}}</td></tr>`).join('')}}</table></div>`}}
function fieldTable(xs){{return `<div class='scroll'><table><tr><th>字段</th><th>端点/方向</th><th>原始值</th></tr>${{xs.map(x=>`<tr><td>${{esc(x.field)}}</td><td>${{esc(x.endpoint)}}</td><td><pre>${{pretty(x.value)}}</pre></td></tr>`).join('')}}</table></div>`}}
function featureRows(r){{let fine=r.explainable_features.detail_explanations||[],parents=r.explainable_features.semantic_parent_explanations||[];return [...fine.map(x=>({{...x,kind:'细粒度特征'}})),...parents.map(x=>({{...x,kind:'语义父节点'}}))].map(x=>`<div class='feature'><span class='pill'>${{esc(x.kind)}}</span><code>${{esc(x.token)}}</code><br><span>${{esc(x.meaning)}}</span></div>`).join('')}}
function neighborRows(xs){{return xs.map(n=>`<details><summary>${{esc(n.case_id)}} · 标签 ${{esc(n.label)}} · 相似度 ${{Number(n.similarity||0).toFixed(4)}}${{n.cross_source?' · 跨来源':''}}</summary><div class='grid'><div><b>共享证据</b><pre>${{esc((n.shared_evidence||[]).join('\\n'))}}</pre></div><div><b>差异证据</b><pre>当前case独有\\n${{esc((n.case_only_evidence||[]).join('\\n'))}}\\n\\n历史case独有\\n${{esc((n.history_only_evidence||[]).join('\\n'))}}</pre></div></div></details>`).join('')||'<p>没有历史近邻记录。</p>'}}
function llmRows(x){{let steps=x.steps||[];return `<p><span class='pill ${{x.llm_output_valid?'good':'warn'}}'>协议校验 ${{x.llm_output_valid?'通过':'未通过'}}</span><span class='pill'>原始结论 ${{esc(x.llm_pre_reconcile_verdict)}}</span><span class='pill'>步骤汇总结论 ${{esc(x.llm_step_aggregate_verdict)}}</span><span class='pill'>终局 ${{esc(x.llm_terminal_verdict)}}</span></p><p class='members'>来源：${{esc(x.source_artifact)}}</p>${{steps.map((s,i)=>`<div class='step ${{x.llm_output_valid?'':'invalid'}}'><b>步骤 ${{i+1}} · ${{esc(s.source)}}</b><br>${{esc(s.statement)}}${{(s.tokens||[]).length?`<br><code>${{esc(s.tokens.join(' · '))}}</code>`:''}}</div>`).join('')}}<details><summary>置信度分解与协议处罚</summary><pre>${{pretty({{confidence:x.confidence,confidence_lower_bound:x.confidence_lower_bound,confidence_breakdown:x.confidence_breakdown,compliance_penalties:x.compliance_penalties}})}}</pre></details><details><summary>补采建议</summary><pre>${{pretty(x.acquisition_recommendations)}}</pre></details>`}}
function auditPanels(r){{return `<div class='grid'><div class='card'><h3>已提取的可解释特征</h3>${{featureRows(r)}}<details><summary>量测质量</summary><pre>${{pretty(r.measurement_quality)}}</pre></details></div><div class='card'><h3>证据图</h3><h4>检索与投票</h4><pre>${{pretty(r.evidence_graph.retrieval_and_vote)}}</pre><details open><summary>因果子图与路径</summary><pre>${{pretty(r.evidence_graph.causal_subgraph)}}</pre></details><details><summary>历史近邻证据对照</summary>${{neighborRows(r.evidence_graph.neighbor_evidence_comparison)}}</details></div></div><div class='card'><h3>大模型逐案分析</h3><p>${{esc(r.llm_analysis.status)}}</p>${{llmRows(r.llm_analysis)}}</div>`}}
function enhanceDetail(){{let r=DATA.cases.find(x=>x.review_id===current),headline=detail.querySelector('.headline');if(r&&headline)headline.insertAdjacentHTML('afterend',auditPanels(r))}}
function renderDetail(){{let r=DATA.cases.find(x=>x.review_id===current);if(!r){{detail.innerHTML='<p>请选择 case</p>';return}}let a=notes[r.review_id]||{{}};detail.innerHTML=`<div class='card headline'><h2>${{esc(r.case_id)}}</h2><span class='pill'>${{esc(r.pattern_id)}} · 同型 ${{r.pattern_case_count}} 条</span><span class='pill'>${{esc(r.month)}}</span><span class='pill'>${{esc(r.source_dataset)}}</span><span class='pill'>${{esc(r.topology_id)}}</span><span class='pill ${{r.label_status==='expert_reviewed'?'good':'warn'}}'>${{esc(r.label_status)}}</span><span class='pill danger'>当前标签 ${{esc(r.current_label)}} / 模型预测 ${{esc(r.model_prediction)}}</span><details><summary>查看同模式 case ID</summary><p class='members'>${{r.pattern_member_case_ids.map(esc).join(' · ')}}</p></details></div><div class='grid'><div class='card'><h3>两个方向的原始 lane 数据</h3>${{laneTable('L1 发送 → L2 接收',r.direction_tables.L1_to_L2)}}${{laneTable('L2 发送 → L1 接收',r.direction_tables.L2_to_L1)}}</div><div class='card'><h3>状态与基础信息</h3><h4>状态位</h4><pre>${{pretty(r.status_snapshot)}}</pre><h4>Case 信息</h4><pre>${{pretty(r.case_metadata)}}</pre></div></div><div class='card'><h3>全部原始字段</h3>${{fieldTable(r.raw_field_rows)}}</div><div class='card'><details><summary>完整原始 case JSON</summary><pre>${{pretty(r.raw_case)}}</pre></details></div><div class='card'><h3>人工打标</h3><div class='form'><label>复核结论<select data-k='decision'><option value=''>请选择</option><option>保留当前标签</option><option>当前标签疑似错误</option><option>模型分析错误</option><option>标签与模型分析均有问题</option><option>当前数据不可辨识</option><option>需要补采后决定</option></select></label><label>建议标签<select data-k='proposed_label'><option value=''>暂不修改</option><option>L1</option><option>L2</option><option>fiber</option></select></label><label>人工置信度<select data-k='confidence'><option value=''>请选择</option><option>high</option><option>medium</option><option>low</option></select></label><label>问题类型<select data-k='issue_type'><option value=''>请选择</option><option>标签语义/标注问题</option><option>模型因果规则错误</option><option>证据图检索或投票错误</option><option>跨来源/拓扑负迁移</option><option>特征漏提取</option><option>原始数据缺失或无效</option><option>fiber能力缺失</option><option>当前快照不可辨识</option></select></label><label class='wide'>决定性原始证据<textarea data-k='decisive_evidence' placeholder='填写字段、端点、lane 和原始值'></textarea></label><label class='wide'>缺失证据/建议补采<textarea data-k='missing_evidence'></textarea></label><label class='wide'>审核备注<textarea data-k='notes'></textarea></label><label><input type='checkbox' data-k='completed'> 已完成复核</label></div></div>`;detail.querySelectorAll('[data-k]').forEach(el=>{{let k=el.dataset.k;el.type==='checkbox'?el.checked=!!a[k]:el.value=a[k]||'';el.onchange=el.oninput=()=>save(k,el.type==='checkbox'?el.checked:el.value)}})}}
function save(k,v){{notes[current]=notes[current]||{{}};notes[current][k]=v;localStorage.setItem(KEY,JSON.stringify(notes));updateProgress();renderList()}}function pick(id){{current=id;renderList();window.scrollTo(0,0)}}function updateProgress(){{progress.textContent=`${{DATA.cases.filter(r=>notes[r.review_id]?.completed).length}} / ${{DATA.cases.length}} 已复核`}}function exportNotes(){{let blob=new Blob([JSON.stringify({{schema:'aggregated-badcase-human-annotations-v1',annotations:notes}},null,2)],{{type:'application/json'}}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='aggregated_badcase_annotations.json';a.click();URL.revokeObjectURL(a.href)}}function importNotes(e){{let f=e.target.files[0];if(!f)return;let rd=new FileReader();rd.onload=()=>{{let x=JSON.parse(rd.result);notes=x.annotations||x;localStorage.setItem(KEY,JSON.stringify(notes));renderList()}};rd.readAsText(f)}}renderList();
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--llm-analysis", type=Path, default=DEFAULT_LLM_ANALYSIS)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    input_reviews = list(payload["reviews"])
    llm_rows = json.loads(args.llm_analysis.read_text(encoding="utf-8"))
    llm_by_case = {str(row["case_id"]): row for row in llm_rows}
    source_badcases = [
        row for row in input_reviews
        if row.get("model_prediction") != row.get("truth_label")
    ]
    llm_steps_eligible_reviews = [
        row for row in source_badcases
        if has_nonempty_llm_steps(llm_by_case.get(str(row["case_id"])))
    ]
    feature_eligible_reviews = [
        row for row in llm_steps_eligible_reviews if has_explainable_features(row)
    ]
    quality_eligible_reviews = [
        row for row in feature_eligible_reviews
        if has_direct_llm_explanation(llm_by_case.get(str(row["case_id"])))
    ]
    filtered_no_llm_steps_count = len(source_badcases) - len(llm_steps_eligible_reviews)
    filtered_no_explainable_features_count = (
        len(llm_steps_eligible_reviews) - len(feature_eligible_reviews)
    )
    filtered_no_direct_llm_explanation_count = (
        len(feature_eligible_reviews) - len(quality_eligible_reviews)
    )
    cases, patterns = build_cases(input_reviews, llm_by_case, args.llm_analysis)
    assert all(row["llm_analysis"].get("steps") for row in cases)
    assert all(has_explainable_features(row) for row in cases)
    assert all(has_direct_llm_explanation(row["llm_analysis"]) for row in cases)
    summary = {
        "schema": SCHEMA,
        "source_experiment": "20260829_snapshot-l1l2-evolution monthly prequential evaluation",
        "source_badcase_count": len(source_badcases),
        "llm_steps_eligible_source_badcase_count": len(llm_steps_eligible_reviews),
        "filtered_no_llm_steps_count": filtered_no_llm_steps_count,
        "filtered_no_explainable_features_count": filtered_no_explainable_features_count,
        "filtered_no_direct_llm_explanation_count": filtered_no_direct_llm_explanation_count,
        "quality_eligible_source_badcase_count": len(quality_eligible_reviews),
        "selected_case_count": len(cases),
        "pattern_count": len(patterns),
        "filtered_duplicate_case_count": len(quality_eligible_reviews) - len(cases),
        "selection_policy": "require explainable features and a nonempty direct LLM explanation, then select one representative per observable error/evidence pattern",
        "source_distribution": dict(Counter(row["source_dataset"] for row in cases)),
        "label_status_distribution": dict(Counter(row["label_status"] for row in cases)),
        "failure_layer_distribution": dict(Counter(row["failure_layer"] for row in cases)),
        "llm_analysis_source": str(args.llm_analysis),
        "llm_record_count": sum(bool(row["llm_analysis"]["available"]) for row in cases),
        "llm_valid_output_count": sum(row["llm_analysis"].get("llm_output_valid") is True for row in cases),
        "llm_nonempty_steps_count": sum(bool(row["llm_analysis"].get("steps")) for row in cases),
        "automatic_relabel": False,
        "n8_frozen": True,
        "reference_only": True,
    }
    template = {
        "schema": "aggregated-badcase-human-annotations-v1",
        "annotations": {row["review_id"]: {
            "case_id": row["case_id"], "pattern_id": row["pattern_id"],
            "decision": "", "proposed_label": "", "confidence": "",
            "issue_type": "", "decisive_evidence": "", "missing_evidence": "",
            "notes": "", "completed": False,
        } for row in cases},
    }

    args.output.mkdir(parents=True, exist_ok=True)
    case_output = args.output / "cases"
    case_output.mkdir(parents=True, exist_ok=True)
    for stale_case in case_output.glob("*.json"):
        stale_case.unlink()
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "patterns.json", {"schema": SCHEMA, "patterns": patterns})
    write_json(args.output / "selected_badcases.json", {"schema": SCHEMA, "summary": summary, "cases": cases})
    write_json(args.output / "annotation_template.json", template)
    for row in cases:
        write_json(case_output / f"{row['case_id']}.json", row)
    (args.output / "badcase_annotation_workbench.html").write_text(render_html(cases, summary), encoding="utf-8")
    (args.output / "README.md").write_text(
        "# 聚合 Bad Case 审计标注工作台\n\n"
        f"源实验共有 {len(source_badcases)} 条冻结错误预测；先剔除 "
        f"{filtered_no_llm_steps_count} 条没有非空历史LLM分析步骤的case，剩余 "
        f"{len(llm_steps_eligible_reviews)} 条中再剔除 {filtered_no_explainable_features_count} 条没有"
        f"可解释特征、{filtered_no_direct_llm_explanation_count} 条没有直接LLM解释的case。最终 "
        f"{len(quality_eligible_reviews)} 条按错误层、来源/拓扑、混淆方向、"
        f"SOP分支、终局路径和可见证据族聚合为 {len(patterns)} 组，每组保留一个代表case，"
        f"过滤 {len(quality_eligible_reviews) - len(cases)} 条重复模式。页面展示原始case、冻结标签/预测、"
        "同模式case ID、可解释特征、证据图和历史冻结LLM逐步分析，并保留人工标注表；不提供逐case"
        "入选解释。该旧实验仅作参考，不回灌活动知识。\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
