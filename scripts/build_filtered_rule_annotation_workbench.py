#!/usr/bin/env python3
"""Build an auditable train/test label-conflict annotation workbench.

The ranking uses frozen train-only features.  Test labels are revealed only after
retrieval and are used exclusively to assemble human-review queues; nothing is
written back to the dataset or offline knowledge bundle.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca_framework.data import cases_by_manifest_split  # noqa: E402
from rca_framework.evidence_graph import match_many  # noqa: E402
from rca_framework.knowledge import OfflineKnowledgeBundle  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "datasets/filtered_rule_temporal_2025_06_09_v1"
DEFAULT_KNOWLEDGE = ROOT / "artifacts/filtered_rule_deterministic_knowledge_v1/knowledge_bundle.json"
DEFAULT_EVALUATION = ROOT / "artifacts/current_model_case_review_v1/blind_evaluation.json"
DEFAULT_OUTPUT = ROOT / "artifacts/filtered_rule_label_annotation_v1"
LABELS = ("L1", "L2", "fiber")


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compact_case(case: Mapping[str, Any]) -> Dict[str, Any]:
    omit = {"label"}
    return {key: value for key, value in case.items() if key not in omit}


def evaluation_index(path: Path) -> tuple[Dict[str, Any], Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cold: Dict[str, Any] = {}
    expert: Dict[str, Any] = {}
    for dataset in payload["datasets"]:
        cold.update({row["case_id"]: row for row in dataset["rows"]})
    for dataset in payload["expert_datasets"]:
        expert.update({row["case_id"]: row for row in dataset["rows"]})
    return cold, expert


def bad_case_experience(expert: Mapping[str, Any]) -> Dict[str, Any]:
    rows = [row for row in expert.values() if row["dataset"] == "all_data" and not row["correct"]]
    by_rule = Counter(row["rule"] for row in rows)
    confusion = Counter(f"{row['actual']}->{row['verdict']}" for row in rows)
    findings = [
        {
            "id": "E1_NO_ANOMALY_IS_ABSTENTION",
            "title": "无异常不等于本端根因",
            "support": by_rule["no_anomaly_default_L1"],
            "knowledge": "固定阈值没有检出异常时，应输出证据不足/补采瞬态数据，不应默认投 L1。",
            "action": "把 no_anomaly 作为降级条件；保留告警端、时间窗口和缺失字段供人工复核。",
        },
        {
            "id": "E2_DIRECTION_RULE_NEEDS_CAUSAL_GUARD",
            "title": "对端异常到本端根因的方向映射需要因果门禁",
            "support": sum(by_rule[name] for name in ("serdes_media_rx_combination", "serdes_snr", "rxpower", "media_snr")),
            "knowledge": "单侧质量劣化既可能是对端发送问题，也可能是本端接收、介质或瞬态问题；不能只按异常出现侧做反向定界。",
            "action": "要求 Tx/Bias、Rx/LOS、跨端 transmission 和同 lane 一致性组成方向证据链。",
        },
        {
            "id": "E3_PORT_STATUS_IS_EVENT_CONTEXT",
            "title": "端口 Down 是事件上下文，不是充分根因",
            "support": by_rule["port_status_L1_down"] + by_rule["port_status_L2_down"],
            "knowledge": "端口状态可能是故障结果、保护动作或采集时刻差异，单独短路会覆盖更细粒度物理证据。",
            "action": "端口状态只提高告警严重度；除非与发送关闭或接收失光同侧一致，否则不直接终裁。",
        },
        {
            "id": "E4_FIBER_REQUIRES_POSITIVE_CHAIN",
            "title": "fiber 需要双向介质证据链",
            "support": sum(1 for row in rows if row["actual"] == "fiber"),
            "knowledge": "同优先级相反定界只是冲突，不是 fiber 的充分证据；fiber 需要两端发送健康而跨端接收衰减/失光等正向证据。",
            "action": "缺少双向正证据时保留 fiber 候选并降级，不以规则冲突直接确诊。",
        },
        {
            "id": "E5_SIMILARITY_CONFLICT_IS_REVIEW_SIGNAL",
            "title": "高相似异标签是标注复核信号，不是自动改标规则",
            "support": 0,
            "knowledge": "相同或近似可见证据可能对应不同瞬态阶段，也可能是标签问题；必须结合原始遥测和缺失证据人工判断。",
            "action": "按冲突组复核主 case 与全部近邻，禁止由近邻多数标签自动回写测试标签。",
        },
    ]
    return {
        "schema_version": "filtered-rule-all-data-badcase-experience-v1",
        "case_count": len(rows),
        "rule_distribution": dict(by_rule.most_common()),
        "confusion_distribution": dict(confusion.most_common()),
        "findings": findings,
    }


def build_groups(data_dir: Path, bundle: OfflineKnowledgeBundle, cold: Mapping[str, Any], expert: Mapping[str, Any], threshold: float) -> list[Dict[str, Any]]:
    train = cases_by_manifest_split(data_dir, "train")
    train_by_id = {row["case_id"]: row for row in train}
    groups: list[Dict[str, Any]] = []
    for split in ("test/all_data", "test/rule1_channel_not_4"):
        tests = cases_by_manifest_split(data_dir, split)
        _, features = bundle.extract_test_features(tests)
        matches = match_many(bundle.graph, features, top_k=0)
        for test, feature, result in zip(tests, features, matches):
            candidates = [
                c for c in result.retrieval_candidates
                if min(c.feature_similarity, c.graph_similarity) >= threshold and c.label != test["label"]
            ]
            if not candidates:
                continue
            candidates.sort(key=lambda c: (-min(c.feature_similarity, c.graph_similarity), -(c.feature_similarity + c.graph_similarity), c.case_id))
            best = candidates[0]
            members = []
            for c in candidates[:8]:
                source = train_by_id[c.case_id]
                members.append({
                    "case_id": c.case_id, "role": "train_neighbor", "label": c.label,
                    "source_dataset": c.source_dataset, "topology_id": c.topology_id,
                    "feature_similarity": c.feature_similarity, "graph_similarity": c.graph_similarity,
                    "shared_evidence": list(c.shared_evidence), "test_only_evidence": list(c.extra_evidence),
                    "train_only_evidence": list(c.missing_evidence),
                    "conflicting_evidence": [list(x) for x in c.conflicting_evidence],
                    "raw_case": compact_case(source),
                })
            expert_row = expert.get(test["case_id"], {})
            cold_row = cold.get(test["case_id"], {})
            score = min(best.feature_similarity, best.graph_similarity)
            exact = best.feature_similarity == 1.0 and best.graph_similarity == 1.0
            priority = round(100 * score + (20 if exact else 0) + (12 if expert_row and not expert_row.get("correct") else 0) + (8 if test["label"] == "fiber" else 0), 2)
            groups.append({
                "group_id": f"conflict_{test['case_id']}", "priority": priority,
                "conflict_type": "exact" if exact else "near", "split": split,
                "main_case": {"case_id": test["case_id"], "role": "test_main", "label": test["label"],
                    "source_dataset": feature.source_dataset, "topology_id": feature.topology_id,
                    "tokens": list(feature.tokens), "raw_case": compact_case(test),
                    "cold_prediction": {k: cold_row.get(k) for k in ("verdict", "confidence", "reasoning", "correct")},
                    "expert_prediction": {k: expert_row.get(k) for k in ("verdict", "rule", "priority", "reasoning", "correct")}},
                "neighbors": members,
                "best_feature_similarity": best.feature_similarity,
                "best_graph_similarity": best.graph_similarity,
                "label_pair": f"{test['label']} vs {best.label}",
            })
    groups.sort(key=lambda row: (-row["priority"], row["group_id"]))
    return groups


def render_html(groups: Sequence[Mapping[str, Any]], experience: Mapping[str, Any]) -> str:
    payload = json.dumps({"groups": groups, "experience": experience}, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Filtered-rule 标签冲突标注工作台</title><style>
:root{{--bg:#f5f7fb;--panel:#fff;--ink:#172033;--muted:#667085;--line:#d9dfeb;--brand:#2457d6;--bad:#b42318;--ok:#067647}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,-apple-system,sans-serif}}header{{position:sticky;top:0;z-index:5;background:#101828;color:#fff;padding:14px 22px;display:flex;gap:18px;align-items:center;flex-wrap:wrap}}header h1{{font-size:18px;margin:0}}button,select,input,textarea{{font:inherit}}button{{border:0;border-radius:7px;padding:8px 12px;cursor:pointer}}.primary{{background:var(--brand);color:#fff}}.ghost{{background:#e9eefb;color:#163a8c}}main{{display:grid;grid-template-columns:330px 1fr;min-height:calc(100vh - 62px)}}aside{{border-right:1px solid var(--line);background:#fff;padding:14px;overflow:auto;height:calc(100vh - 62px);position:sticky;top:62px}}.content{{padding:20px;max-width:1500px}}.filters{{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px}}.filters input{{grid-column:1/-1}}input,select,textarea{{border:1px solid #cbd3e1;border-radius:6px;padding:7px;background:#fff}}.item{{padding:10px;border:1px solid var(--line);border-radius:8px;margin:7px 0;cursor:pointer}}.item.active{{border:2px solid var(--brand);background:#f1f5ff}}.item.done{{opacity:.58}}.pill{{display:inline-block;border-radius:999px;padding:2px 7px;background:#eef2f8;margin:2px;font-size:12px}}.danger{{background:#fee4e2;color:var(--bad)}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:14px;box-shadow:0 1px 2px #1018280a}}h2,h3{{margin:0 0 10px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.metrics{{display:flex;gap:10px;flex-wrap:wrap}}.metric{{padding:8px 12px;background:#f2f4f7;border-radius:7px}}details{{border-top:1px solid var(--line);padding:8px 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#101828;color:#e6edf8;padding:12px;border-radius:7px;max-height:430px;overflow:auto}}table{{width:100%;border-collapse:collapse}}th,td{{border:1px solid var(--line);padding:6px;text-align:left;vertical-align:top}}textarea{{width:100%;min-height:80px}}.form{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}.wide{{grid-column:1/-1}}.notice{{background:#fffaeb;border-color:#fedf89}}@media(max-width:900px){{main{{display:block}}aside{{position:static;height:auto}}.grid,.form{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>标签冲突标注工作台</h1><span id='progress'></span><button class='primary' onclick='exportAnnotations()'>导出标注 JSON</button><button class='ghost' onclick='document.getElementById("importer").click()'>导入</button><input id='importer' type='file' hidden accept='.json' onchange='importAnnotations(event)'></header>
<main><aside><div class='filters'><input id='search' placeholder='搜索 case / 标签 / 规则' oninput='renderList()'><select id='split' onchange='renderList()'><option value=''>全部数据集</option><option>test/all_data</option><option>test/rule1_channel_not_4</option></select><select id='kind' onchange='renderList()'><option value=''>精确+近似</option><option value='exact'>精确冲突</option><option value='near'>近似冲突</option></select><select id='state' onchange='renderList()'><option value=''>全部状态</option><option value='open'>未完成</option><option value='done'>已完成</option></select></div><div id='list'></div></aside><section class='content'><div id='detail'></div></section></main>
<script id='dataset' type='application/json'>{payload}</script><script>
const DATA=JSON.parse(document.getElementById('dataset').textContent), KEY='filtered_rule_annotations_v1';let notes=JSON.parse(localStorage.getItem(KEY)||'{{}}'), current=DATA.groups[0]?.group_id;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
function filtered(){{let q=search.value.toLowerCase();return DATA.groups.filter(g=>(!split.value||g.split===split.value)&&(!kind.value||g.conflict_type===kind.value)&&(!state.value||(state.value==='done')===!!notes[g.group_id]?.completed)&&JSON.stringify(g).toLowerCase().includes(q))}}
function renderList(){{let rows=filtered();list.innerHTML=rows.map(g=>`<div class="item ${{g.group_id===current?'active':''}} ${{notes[g.group_id]?.completed?'done':''}}" onclick="selectGroup('${{g.group_id}}')"><b>${{esc(g.main_case.case_id)}}</b><br><span class="pill ${{g.conflict_type==='exact'?'danger':''}}">${{g.conflict_type}}</span><span class="pill">${{esc(g.label_pair)}}</span><span class="pill">${{g.best_feature_similarity.toFixed(3)}} / ${{g.best_graph_similarity.toFixed(3)}}</span><br><small>${{esc(g.split)}} · priority ${{g.priority}}</small></div>`).join('')||'<p>无匹配项</p>';updateProgress()}}
function selectGroup(id){{current=id;renderList();renderDetail()}}
function pretty(v){{return esc(JSON.stringify(v,null,2))}}
function tokens(n){{return `<details><summary>${{esc(n.case_id)}} · ${{esc(n.label)}} · S=${{n.feature_similarity.toFixed(3)}}/${{n.graph_similarity.toFixed(3)}}</summary><div class=grid><div><b>共享证据</b><pre>${{esc(n.shared_evidence.join('\n'))}}</pre></div><div><b>差异/冲突</b><pre>测试独有:\n${{esc(n.test_only_evidence.join('\n'))}}\n\n训练独有:\n${{esc(n.train_only_evidence.join('\n'))}}\n\n互斥:\n${{pretty(n.conflicting_evidence)}}</pre></div></div><details><summary>原始训练 case</summary><pre>${{pretty(n.raw_case)}}</pre></details></details>`}}
function renderDetail(){{let g=DATA.groups.find(x=>x.group_id===current);if(!g){{detail.innerHTML='<p>请选择一个冲突组</p>';return}}let a=notes[g.group_id]||{{}};detail.innerHTML=`<div class="card notice"><h2>${{esc(g.main_case.case_id)}} <span class="pill danger">${{esc(g.label_pair)}}</span></h2><div class=metrics><span class=metric>特征相似 ${{g.best_feature_similarity.toFixed(4)}}</span><span class=metric>图相似 ${{g.best_graph_similarity.toFixed(4)}}</span><span class=metric>${{g.neighbors.length}} 个异标签近邻</span><span class=metric>${{esc(g.conflict_type)}}</span></div><p>这是人工复核队列，不代表标签自动错误。测试标签未参与特征构建和近邻排序。</p></div><div class=grid><div class=card><h3>测试主 case</h3><p><b>现标签：</b>${{esc(g.main_case.label)}}　<b>来源：</b>${{esc(g.main_case.source_dataset)}}</p><details open><summary>冷启动与专家判断</summary><pre>${{pretty({{cold:g.main_case.cold_prediction,expert:g.main_case.expert_prediction}})}}</pre></details><details><summary>可解释特征</summary><pre>${{esc(g.main_case.tokens.join('\n'))}}</pre></details><details><summary>原始遥测</summary><pre>${{pretty(g.main_case.raw_case)}}</pre></details></div><div class=card><h3>训练近邻与差异</h3>${{g.neighbors.map(tokens).join('')}}</div></div><div class=card><h3>人工标注</h3><div class=form><label>冲突结论<select data-k=decision><option value=''>请选择</option><option>测试标签疑似错误</option><option>训练标签疑似错误</option><option>两者均合理（证据不可辨识）</option><option>两者均需复核</option><option>当前证据不足</option></select></label><label>建议标签<select data-k=proposed_label><option value=''>不修改</option>${{['L1','L2','fiber'].map(x=>`<option>${{x}}</option>`)}}</select></label><label>置信度<select data-k=confidence><option value=''>请选择</option><option>high</option><option>medium</option><option>low</option></select></label><label>问题类型<select data-k=issue_type><option value=''>请选择</option><option>标签问题</option><option>瞬态/时间错位</option><option>可见特征不可辨识</option><option>数据缺失</option><option>特征抽取缺陷</option><option>物理规则缺陷</option></select></label><label>证据充分性<select data-k=evidence_sufficiency><option value=''>请选择</option><option>充分</option><option>部分充分</option><option>不足</option></select></label><label>审核人<input data-k=reviewer></label><label class=wide>决定性证据<textarea data-k=decisive_evidence placeholder='明确指出端、方向、指标、lane 和数值'></textarea></label><label class=wide>缺失证据/建议补采<textarea data-k=missing_evidence></textarea></label><label class=wide>审核备注<textarea data-k=notes></textarea></label><label><input type=checkbox data-k=completed> 已完成复核</label></div></div>`;detail.querySelectorAll('[data-k]').forEach(el=>{{let k=el.dataset.k;el.type==='checkbox'?el.checked=!!a[k]:el.value=a[k]||'';el.onchange=el.oninput=()=>save(k,el.type==='checkbox'?el.checked:el.value)}})}}
function save(k,v){{notes[current]={{...(notes[current]||{{}}),[k]:v,updated_at:new Date().toISOString()}};localStorage.setItem(KEY,JSON.stringify(notes));renderList()}}
function updateProgress(){{let done=DATA.groups.filter(g=>notes[g.group_id]?.completed).length;progress.textContent=`已完成 ${{done}} / ${{DATA.groups.length}}`}}
function exportAnnotations(){{let blob=new Blob([JSON.stringify({{schema_version:'filtered-rule-human-annotations-v1',exported_at:new Date().toISOString(),annotations:notes}},null,2)],{{type:'application/json'}}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='filtered_rule_human_annotations.json';a.click();URL.revokeObjectURL(a.href)}}
function importAnnotations(e){{let f=e.target.files[0];if(!f)return;let r=new FileReader;r.onload=()=>{{let x=JSON.parse(r.result);notes=x.annotations||x;localStorage.setItem(KEY,JSON.stringify(notes));renderList();renderDetail()}};r.readAsText(f)}}
renderList();renderDetail();
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--knowledge", type=Path, default=DEFAULT_KNOWLEDGE)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--similarity-threshold", type=float, default=0.70)
    args = parser.parse_args()
    bundle = OfflineKnowledgeBundle.load(args.knowledge)
    cold, expert = evaluation_index(args.evaluation)
    experience = bad_case_experience(expert)
    groups = build_groups(args.data_dir, bundle, cold, expert, args.similarity_threshold)
    experience["findings"][-1]["support"] = len(groups)
    summary = {
        "schema_version": "filtered-rule-label-conflict-review-v1", "group_count": len(groups),
        "exact_group_count": sum(g["conflict_type"] == "exact" for g in groups),
        "near_group_count": sum(g["conflict_type"] == "near" for g in groups),
        "all_data_group_count": sum(g["split"] == "test/all_data" for g in groups),
        "rule1_group_count": sum(g["split"] == "test/rule1_channel_not_4" for g in groups),
        "similarity_threshold": args.similarity_threshold, "knowledge_bundle_hash": bundle.content_hash(),
        "test_labels_used_for": "post-retrieval human-review queue only", "n8_frozen": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dump(args.output_dir / "conflict_groups.json", {"summary": summary, "groups": groups})
    dump(args.output_dir / "all_data_bad_case_experience.json", experience)
    dump(args.output_dir / "annotation_template.json", {"schema_version": "filtered-rule-human-annotations-v1", "annotations": {}})
    (args.output_dir / "annotation_workbench.html").write_text(render_html(groups, experience), encoding="utf-8")
    readme = ["# Filtered-rule 标签冲突复核", "", "本目录只生成测试标签人工复核队列，不修改数据、知识包或模型。", "", f"- 冲突组：{len(groups)}（精确 {summary['exact_group_count']}，近似 {summary['near_group_count']}）", f"- all_data bad case：{experience['case_count']}", f"- 相似度门槛：特征和语义图的较小值 >= {args.similarity_threshold}", "- 打开 `annotation_workbench.html`，标注自动保存在浏览器 localStorage，并可导入/导出 JSON。", ""]
    (args.output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
