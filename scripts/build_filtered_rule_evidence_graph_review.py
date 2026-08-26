#!/usr/bin/env python3
"""Render a standalone, interactive review of the filtered-rule evidence graph."""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca_framework.knowledge import OfflineKnowledgeBundle  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE = ROOT / "artifacts/filtered_rule_deterministic_knowledge_v2/knowledge_bundle.json"
OUTPUT = ROOT / "artifacts/filtered_rule_evidence_graph_review_v2"

NODE_EXPLANATIONS = {
    "Case": ("一次经过标准化的历史故障事件，是证据图中的检索主体。", "一张历史故障工单。"),
    "FeatureToken": ("由冻结特征字典生成的离散物理证据维度；检索只读取这些节点，不读取标签。", "这条故障表现出了什么现象。"),
    "SOPStep": ("确定性训练流程记录的证据链步骤，包含路由、约束上下文、统计树路径或专家SOP上下文。", "系统当时按什么顺序分析。"),
    "ConstraintCheck": ("与某一步显式绑定的可执行物理约束检查。", "这一步有没有经过物理规则核对。"),
    "Outcome": ("训练期分支与决策门禁的输出，同时保留人工/数据集确认标签用于审计。", "系统结论和已知正确答案。"),
}

EDGE_EXPLANATIONS = {
    "has_token": ("Case到FeatureToken的事实边，是IDF-Jaccard检索的主要输入。", "这张工单出现了这个现象。"),
    "has_step": ("Case到SOPStep的归属边。", "这一步属于这张工单。"),
    "precedes": ("SOP步骤之间的有序边。", "先做前一步，再做后一步。"),
    "uses_token": ("某推理步骤实际引用当前case特征的证据边。", "这一步用到了这个现象。"),
    "supports_decision": ("步骤对Outcome提供支持的边；是否足够终裁仍由N6控制。", "这一步支持最终判断。"),
    "checked_by": ("步骤到约束检查的绑定边。", "这一步交给哪条规则检查。"),
    "constrains_decision": ("约束检查对Outcome施加排除或一致性限制。", "这条规则限制最终结论。"),
    "concludes": ("Case到Outcome的最终关联边。", "这张工单最终得到什么判断。"),
}


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def family(token: str) -> str:
    return token.split(":", 1)[0]


def analyze(bundle: OfflineKnowledgeBundle) -> Dict[str, Any]:
    graph = bundle.graph
    diagnosis_by_id = {row.case_id: row for row in graph.case_diagnoses}
    token_cases: Dict[str, list[Any]] = defaultdict(list)
    signatures: Dict[tuple[str, ...], list[Any]] = defaultdict(list)
    for case in graph.cases:
        signatures[case.tokens].append(case)
        for token in case.tokens:
            token_cases[token].append(case)

    token_rows = []
    for token, cases in token_cases.items():
        labels = Counter(case.label for case in cases)
        sources = Counter(case.source_dataset for case in cases)
        token_rows.append({
            "token": token, "family": family(token), "support": len(cases),
            "idf": graph.idf.get(token, 0.0), "labels": dict(labels), "sources": dict(sources),
            "purity": max(labels.values()) / len(cases), "case_ids": [case.case_id for case in cases],
        })
    token_rows.sort(key=lambda row: (-row["support"], row["token"]))

    signature_rows = []
    for tokens, cases in signatures.items():
        labels = Counter(case.label for case in cases)
        signature_rows.append({
            "signature_id": f"sig-{len(signature_rows)+1:03d}", "support": len(cases),
            "pure": len(labels) == 1, "labels": dict(labels), "tokens": list(tokens),
            "case_ids": [case.case_id for case in cases],
        })
    signature_rows.sort(key=lambda row: (-row["support"], row["signature_id"]))

    node_types: Counter[str] = Counter()
    edge_types: Counter[str] = Counter()
    step_kinds: Counter[str] = Counter()
    branch_counts: Counter[str] = Counter()
    prediction_matches = Counter()
    outcome_verdict_count = 0
    outcome_correct_count = 0
    diagnosis_rows = []
    for case in graph.cases:
        diagnosis = diagnosis_by_id.get(case.case_id)
        if diagnosis is None:
            continue
        nodes = [node.to_dict() for node in diagnosis.nodes]
        edges = [edge.to_dict() for edge in diagnosis.edges]
        node_types.update(node["type"] for node in nodes)
        edge_types.update(edge["type"] for edge in edges)
        outcome = next((node for node in nodes if node["type"] == "Outcome"), {"attrs": {}})["attrs"]
        branch_counts[str(outcome.get("branch", "unknown"))] += 1
        if outcome.get("verdict"):
            outcome_verdict_count += 1
            outcome_correct_count += int(bool(outcome.get("prediction_matches_confirmation")))
            prediction_matches[str(bool(outcome.get("prediction_matches_confirmation")))] += 1
        else:
            prediction_matches["no_verdict"] += 1
        for node in nodes:
            if node["type"] == "SOPStep":
                step_kinds[str(node["attrs"].get("kind", "unknown"))] += 1
        diagnosis_rows.append({
            "case_id": case.case_id, "label": case.label, "source_dataset": case.source_dataset,
            "topology_id": case.topology_id, "tokens": list(case.tokens), "nodes": nodes, "edges": edges,
            "outcome": outcome,
        })

    case_token_edges = sum(len(case.tokens) for case in graph.cases)
    isolated_tokens = sum(row["support"] == 1 for row in token_rows)
    mixed = [row for row in signature_rows if not row["pure"]]
    findings = [
        {"status": "OK", "title": "检索路径与标签读取隔离", "detail": "倒排索引和相似度只使用Case—FeatureToken边；历史label挂在Case上，只有N5a候选复用时显式读取。"},
        {"status": "RISK", "title": "历史图仍然稀疏", "detail": f"124个case形成{len(signature_rows)}个signature，{sum(r['support']==1 for r in signature_rows)}个只有单条支持；图更像案例索引，不是高复用规则图。"},
        {"status": "RISK", "title": "混合标签signature必须禁止自动复用", "detail": f"{len(mixed)}个混合标签signature覆盖{sum(r['support'] for r in mixed)}条训练case，应进入冲突仲裁而不是N5a直接沿用。"},
        {"status": "RISK", "title": "诊断子图中的约束多数是上下文描述", "detail": f"124个诊断子图中只有{node_types['ConstraintCheck']}个ConstraintCheck节点；大部分物理约束只写在SOP上下文文本中，没有形成可执行逐条检查边。"},
        {"status": "RISK", "title": "训练诊断子图不能直接当作已确认排障经验", "detail": f"124个Outcome中只有{outcome_verdict_count}个包含自动verdict，{124-outcome_verdict_count}个是降级/空结论；有结论部分{outcome_correct_count}/{outcome_verdict_count or 1}与训练确认标签一致。当前子图主要是流程trace，不是124条完整的已确认因果链。"},
        {"status": "REVISE", "title": "统计树步骤不应伪装为物理证据", "detail": f"numeric_decision_tree步骤{step_kinds['numeric_decision_tree']}个，只能作为训练统计先验，建议在图中单独标为StatisticalPrior节点。"},
    ]
    return {
        "schema_version": "filtered-rule-evidence-graph-review-v2",
        "graph_version": graph.version, "graph_schema": graph.schema_version,
        "graph_hash": graph.content_hash(), "dictionary_version": graph.dictionary_version,
        "dictionary_hash": graph.dictionary_hash, "knowledge_bundle_hash": bundle.content_hash(),
        "case_count": len(graph.cases), "token_count": len(token_rows), "case_token_edge_count": case_token_edges,
        "diagnosis_count": len(diagnosis_rows), "diagnosis_node_count": sum(node_types.values()),
        "diagnosis_edge_count": sum(edge_types.values()), "node_types": dict(node_types),
        "edge_types": dict(edge_types), "step_kinds": dict(step_kinds), "branch_counts": dict(branch_counts),
        "prediction_matches_confirmation": dict(prediction_matches),
        "outcome_verdict_count": outcome_verdict_count,
        "outcome_no_verdict_count": len(graph.cases) - outcome_verdict_count,
        "outcome_correct_count": outcome_correct_count,
        "label_distribution": graph.label_distribution(),
        "source_distribution": dict(Counter(case.source_dataset for case in graph.cases)),
        "topology_distribution": dict(Counter(case.topology_id for case in graph.cases)),
        "signature_count": len(signature_rows), "singleton_signature_count": sum(r["support"] == 1 for r in signature_rows),
        "mixed_signature_count": len(mixed), "mixed_signature_case_count": sum(r["support"] for r in mixed),
        "singleton_token_count": isolated_tokens, "token_rows": token_rows,
        "signature_rows": signature_rows, "diagnosis_rows": diagnosis_rows,
        "node_explanations": {key: {"professional": value[0], "plain": value[1]} for key, value in NODE_EXPLANATIONS.items()},
        "edge_explanations": {key: {"professional": value[0], "plain": value[1]} for key, value in EDGE_EXPLANATIONS.items()},
        "findings": findings,
    }


def render(report: Mapping[str, Any]) -> str:
    e = html.escape
    compact = {
        "cases": report["diagnosis_rows"], "tokens": report["token_rows"],
        "signatures": report["signature_rows"],
    }
    payload = json.dumps(compact, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    finding_html = "".join(f"<article class='finding {row['status'].lower()}'><b>{e(row['status'])} · {e(row['title'])}</b><p>{e(row['detail'])}</p></article>" for row in report["findings"])
    node_rows = "".join(f"<tr><td><code>{e(name)}</code></td><td>{report['node_types'].get(name,0)}</td><td>{e(pro['professional'])}</td><td>{e(pro['plain'])}</td></tr>" for name, pro in report["node_explanations"].items())
    edge_rows = "".join(f"<tr><td><code>{e(name)}</code></td><td>{report['edge_types'].get(name,0)}</td><td>{e(pro['professional'])}</td><td>{e(pro['plain'])}</td></tr>" for name, pro in report["edge_explanations"].items())
    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Evidence Graph逐层Review</title><style>
:root{{--bg:#f4f6fa;--panel:#fff;--ink:#172033;--muted:#667085;--line:#d8deea;--blue:#2457d6;--green:#067647;--red:#b42318;--amber:#b54708;--purple:#6941c6}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.6 system-ui,-apple-system,sans-serif}}header{{background:linear-gradient(125deg,#101828,#1849a9);color:#fff;padding:34px max(22px,calc((100% - 1400px)/2))}}header h1{{margin:0;font-size:30px}}header p{{max-width:1000px;color:#dce6ff}}main{{max-width:1400px;margin:auto;padding:22px}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px;margin-bottom:16px}}.metrics{{display:grid;grid-template-columns:repeat(6,1fr);gap:9px}}.metric{{background:#f7f9fc;padding:11px;border-radius:8px}}.metric b{{display:block;font-size:22px;color:var(--blue)}}h2{{margin-top:0}}.flow{{display:grid;grid-template-columns:repeat(5,1fr);align-items:center;gap:22px;margin:22px 0}}.flow div{{border:2px solid #b9ccff;background:#f0f5ff;padding:14px;text-align:center;border-radius:9px;position:relative}}.flow div:not(:last-child):after{{content:'→';position:absolute;right:-22px;top:25%;font-size:24px;color:var(--blue)}}.findings{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}.finding{{border-left:4px solid var(--green);padding:10px;background:#f6fef9}}.finding.risk{{border-color:var(--red);background:#fff5f4}}.finding.revise{{border-color:var(--amber);background:#fffaeb}}.finding p{{margin:5px 0}}.table{{overflow:auto}}table{{width:100%;border-collapse:collapse;min-width:850px}}th,td{{border:1px solid var(--line);padding:7px;text-align:left;vertical-align:top}}th{{background:#f2f4f7}}.controls{{display:flex;gap:9px;flex-wrap:wrap;margin-bottom:10px}}input,select{{border:1px solid #cbd3e1;border-radius:7px;padding:7px;font:inherit}}#caseSelect{{min-width:300px}}.graph-wrap{{display:grid;grid-template-columns:minmax(0,2fr) minmax(280px,1fr);gap:12px}}svg{{width:100%;min-height:520px;border:1px solid var(--line);border-radius:9px;background:#fbfcff}}.detail{{background:#f7f9fc;border-radius:9px;padding:12px;overflow-wrap:anywhere}}.detail pre{{white-space:pre-wrap;font-size:12px}}.node{{cursor:pointer}}.node text{{font-size:11px;pointer-events:none}}.edge{{stroke:#98a2b3;stroke-width:1.4;fill:none}}.legend span{{display:inline-flex;align-items:center;margin:4px 10px 4px 0}}.dot{{width:10px;height:10px;border-radius:50%;margin-right:5px}}.bars{{display:grid;gap:5px}}.bar{{display:grid;grid-template-columns:220px 1fr 45px;gap:8px;align-items:center}}.track{{height:12px;background:#eef1f6;border-radius:6px;overflow:hidden}}.fill{{height:100%;background:var(--blue)}}.sig{{display:grid;grid-template-columns:90px 70px 1fr 150px;gap:8px;padding:7px;border-bottom:1px solid var(--line)}}.mixed{{background:#fff5f4}}.hint{{color:var(--muted)}}code{{font-size:12px}}@media(max-width:900px){{.metrics{{grid-template-columns:repeat(3,1fr)}}.flow{{grid-template-columns:1fr}}.flow div:after{{display:none}}.graph-wrap,.findings{{grid-template-columns:1fr}}}}@media(max-width:520px){{.metrics{{grid-template-columns:1fr 1fr}}.bar{{grid-template-columns:130px 1fr 35px}}}}
</style></head><body><header><h1>Filtered-rule v2 证据图逐层 Review</h1><p>从历史Case—FeatureToken检索主图，到每个case的SOP/约束/Outcome诊断子图。图版本 {e(report['graph_version'])}，知识包 {e(report['knowledge_bundle_hash'])}。</p></header><main>
<section class='card'><h2>整体规模</h2><div class='metrics'><div class='metric'><b>{report['case_count']}</b>历史Case</div><div class='metric'><b>{report['token_count']}</b>FeatureToken</div><div class='metric'><b>{report['case_token_edge_count']}</b>Case-token边</div><div class='metric'><b>{report['signature_count']}</b>Signature</div><div class='metric'><b>{report['diagnosis_node_count']}</b>诊断节点</div><div class='metric'><b>{report['outcome_verdict_count']}/{report['case_count']}</b>子图含自动结论</div></div><div class='flow'><div>原始遥测<br><small>不进入图</small></div><div>Case<br><small>124个训练事件</small></div><div>FeatureToken<br><small>IDF-Jaccard检索</small></div><div>SOP / Constraint<br><small>诊断过程trace</small></div><div>Outcome<br><small>仅13个含verdict</small></div></div></section>
<section class='card'><h2>审计结论</h2><div class='findings'>{finding_html}</div></section>
<section class='card'><h2>全图结构分布</h2><div class='graph-wrap'><div><h3>节点类型</h3><div id='nodeBars' class='bars'></div><h3>边类型</h3><div id='edgeBars' class='bars'></div></div><div class='detail'><b>怎么看</b><p>Case—FeatureToken是检索事实层；SOPStep—Outcome是诊断过程层。两层同时存在，但不应把统计树路径或SOP上下文误当成当前case的直接物理证据。</p><p><b>Signature：</b>{report['singleton_signature_count']}/{report['signature_count']} 为单例；混合标签 {report['mixed_signature_count']} 组，覆盖 {report['mixed_signature_case_count']} 条case。</p><p><b>单例token：</b>{report['singleton_token_count']}/{report['token_count']}，低支持token的高纯度不能视为可靠规则。</p></div></div></section>
<section class='card'><h2>逐Case诊断子图</h2><div class='controls'><input id='caseSearch' placeholder='搜索case ID'><select id='caseSelect'></select><select id='caseFilter'><option value=''>全部分支</option><option>N5a</option><option>N5b</option><option>N5c</option></select></div><div class='legend'><span><i class='dot' style='background:#2457d6'></i>Case</span><span><i class='dot' style='background:#12b76a'></i>Feature</span><span><i class='dot' style='background:#f79009'></i>SOPStep</span><span><i class='dot' style='background:#6941c6'></i>Constraint</span><span><i class='dot' style='background:#f04438'></i>Outcome</span></div><div class='graph-wrap'><svg id='caseGraph' viewBox='0 0 900 540' aria-label='case diagnosis graph'></svg><div id='nodeDetail' class='detail'>点击节点查看属性。</div></div></section>
<section class='card'><h2>Signature纯度</h2><div class='controls'><select id='sigFilter'><option value=''>全部</option><option value='mixed'>仅混合标签</option><option value='singleton'>仅单例</option></select></div><div id='sigList'></div></section>
<section class='card'><h2>FeatureToken倒排索引</h2><p class='hint'>support决定IDF；purity只供标签审计。点击token可查看关联case。</p><div class='controls'><input id='tokenSearch' placeholder='搜索token'><select id='familyFilter'><option value=''>全部特征族</option></select></div><div id='tokenTable' class='table'></div></section>
<section class='card'><h2>每种节点的含义</h2><div class='table'><table><thead><tr><th>节点类型</th><th>数量</th><th>专业解释</th><th>通俗解释</th></tr></thead><tbody>{node_rows}</tbody></table></div></section>
<section class='card'><h2>每种边的含义</h2><div class='table'><table><thead><tr><th>边类型</th><th>数量</th><th>专业解释</th><th>通俗解释</th></tr></thead><tbody>{edge_rows}</tbody></table></div></section>
<script id='graphData' type='application/json'>{payload}</script><script>
const D=JSON.parse(document.getElementById('graphData').textContent), nodeCounts={json.dumps(report['node_types'])}, edgeCounts={json.dumps(report['edge_types'])};
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
function bars(id,data){{const m=Math.max(...Object.values(data),1);document.getElementById(id).innerHTML=Object.entries(data).sort((a,b)=>b[1]-a[1]).map(([k,v])=>`<div class=bar><code>${{esc(k)}}</code><div class=track><div class=fill style="width:${{100*v/m}}%"></div></div><b>${{v}}</b></div>`).join('')}}bars('nodeBars',nodeCounts);bars('edgeBars',edgeCounts);
const colors={{Case:'#2457d6',FeatureToken:'#12b76a',SOPStep:'#f79009',ConstraintCheck:'#6941c6',Outcome:'#f04438'}};
function visibleCases(){{let q=caseSearch.value.toLowerCase(),b=caseFilter.value;return D.cases.filter(c=>c.case_id.includes(q)&&(!b||c.outcome.branch===b))}}
function fillCases(){{let rows=visibleCases(),old=caseSelect.value;caseSelect.innerHTML=rows.map(c=>`<option value="${{c.case_id}}">${{c.case_id}} · ${{c.label}} · ${{c.outcome.branch}}</option>`).join('');if(rows.some(c=>c.case_id===old))caseSelect.value=old;drawCase()}}
function layout(nodes){{let groups={{Case:[],FeatureToken:[],SOPStep:[],ConstraintCheck:[],Outcome:[]}};nodes.forEach(n=>(groups[n.type]||groups.SOPStep).push(n));let xs={{Case:80,FeatureToken:285,SOPStep:490,ConstraintCheck:680,Outcome:835}};Object.entries(groups).forEach(([type,rows])=>rows.forEach((n,i)=>{{n.x=xs[type];n.y=60+(420*(i+1)/(rows.length+1))}}));return nodes}}
function drawCase(){{let c=D.cases.find(x=>x.case_id===caseSelect.value)||visibleCases()[0];if(!c){{caseGraph.innerHTML='';return}};let nodes=layout(c.nodes.map(x=>({{...x}}))),by=Object.fromEntries(nodes.map(n=>[n.id,n]));caseGraph.innerHTML=c.edges.filter(x=>by[x.src]&&by[x.dst]).map(x=>`<path class=edge d="M ${{by[x.src].x}} ${{by[x.src].y}} C ${{(by[x.src].x+by[x.dst].x)/2}} ${{by[x.src].y}}, ${{(by[x.src].x+by[x.dst].x)/2}} ${{by[x.dst].y}}, ${{by[x.dst].x}} ${{by[x.dst].y}}"><title>${{esc(x.type)}}</title></path>`).join('')+nodes.map(n=>`<g class=node data-id="${{n.id}}" transform="translate(${{n.x}},${{n.y}})"><circle r="${{n.type==='Outcome'?18:14}}" fill="${{colors[n.type]||'#667085'}}"></circle><text x="${{n.x>700?-20:20}}" y=4 text-anchor="${{n.x>700?'end':'start'}}">${{esc(n.id.length>28?n.id.slice(0,26)+'…':n.id)}}</text></g>`).join('');caseGraph.querySelectorAll('.node').forEach(el=>el.onclick=()=>{{let n=by[el.dataset.id];nodeDetail.innerHTML=`<b>${{esc(n.type)}} · ${{esc(n.id)}}</b><pre>${{esc(JSON.stringify(n.attrs,null,2))}}</pre>`}});nodeDetail.innerHTML=`<b>${{esc(c.case_id)}} · label=${{esc(c.label)}} · ${{esc(c.outcome.branch)}}</b><p>tokens=${{c.tokens.length}}，verdict=${{esc(c.outcome.verdict)}}，matches confirmation=${{esc(c.outcome.prediction_matches_confirmation)}}</p><pre>${{esc(c.tokens.join('\n'))}}</pre>`}}
caseSearch.oninput=fillCases;caseFilter.onchange=fillCases;caseSelect.onchange=drawCase;fillCases();
function drawSigs(){{let f=sigFilter.value,rows=D.signatures.filter(s=>!f||(f==='mixed'?!s.pure:s.support===1));sigList.innerHTML=rows.slice(0,150).map(s=>`<div class="sig ${{s.pure?'':'mixed'}}"><b>${{s.signature_id}}</b><span>support ${{s.support}}</span><code>${{esc(s.tokens.join(' · '))}}</code><span>${{esc(JSON.stringify(s.labels))}}</span></div>`).join('')}}sigFilter.onchange=drawSigs;drawSigs();
const families=[...new Set(D.tokens.map(t=>t.family))].sort();familyFilter.innerHTML+=""+families.map(x=>`<option>${{esc(x)}}</option>`).join('');function drawTokens(){{let q=tokenSearch.value.toLowerCase(),f=familyFilter.value,rows=D.tokens.filter(t=>t.token.toLowerCase().includes(q)&&(!f||t.family===f));tokenTable.innerHTML=`<table><thead><tr><th>token</th><th>family</th><th>support</th><th>IDF</th><th>purity</th><th>labels</th><th>sources</th></tr></thead><tbody>${{rows.map(t=>`<tr title="${{esc(t.case_ids.join(', '))}}"><td><code>${{esc(t.token)}}</code></td><td>${{esc(t.family)}}</td><td>${{t.support}}</td><td>${{t.idf.toFixed(3)}}</td><td>${{(100*t.purity).toFixed(1)}}%</td><td>${{esc(JSON.stringify(t.labels))}}</td><td>${{esc(JSON.stringify(t.sources))}}</td></tr>`).join('')}}</tbody></table>`}}tokenSearch.oninput=drawTokens;familyFilter.onchange=drawTokens;drawTokens();
</script></main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowledge", type=Path, default=KNOWLEDGE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = analyze(OfflineKnowledgeBundle.load(args.knowledge))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dump(args.output_dir / "evidence_graph_review.json", report)
    (args.output_dir / "evidence_graph_review.html").write_text(render(report), encoding="utf-8")
    (args.output_dir / "README.md").write_text(
        "# Filtered-rule v2 Evidence Graph Review\n\n"
        f"- graph: `{report['graph_version']}`\n- cases: {report['case_count']}\n"
        f"- tokens: {report['token_count']}\n- signatures: {report['signature_count']}\n"
        f"- diagnosis nodes/edges: {report['diagnosis_node_count']}/{report['diagnosis_edge_count']}\n",
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in (
        "graph_version", "case_count", "token_count", "case_token_edge_count", "signature_count",
        "singleton_signature_count", "mixed_signature_count", "mixed_signature_case_count",
        "diagnosis_node_count", "diagnosis_edge_count", "node_types", "edge_types", "step_kinds",
        "outcome_verdict_count", "outcome_no_verdict_count", "outcome_correct_count",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
