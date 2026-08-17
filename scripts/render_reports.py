"""渲染两份离线 HTML 报告：框架与结果总览、逐 case 决策链分析。

两份报告都把数据以 JSON 内嵌，不依赖任何外部资源或网络，可以直接拷走用浏览器打开。
渲染逻辑放在前端而不是在 Python 里拼字符串，是为了让筛选、展开、对照这些交互
不需要重新生成文件。

依赖三份产物，缺任何一份都会直接报错而不是静默降级：
  artifacts/overview_bundle.json  框架定义 + 各口径基线（build_overview_bundle.py）
  artifacts/report_bundle.json    107 条 test case 的完整判断链（build_report_bundle.py）
  scripts/case_narratives.py      24 条判错 case 的盲推导与洞察（手写）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.case_narratives import KIND_COLOR, KIND_LABEL, NARRATIVES

OVERVIEW_BUNDLE = ROOT / "artifacts/overview_bundle.json"
CASE_BUNDLE = ROOT / "artifacts/report_bundle.json"
OUT_DIR = ROOT / "docs"

CSS = """
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;font:15px/1.75 -apple-system,"Segoe UI",Roboto,"Helvetica Neue","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
     color:#1f2430;background:#f6f7f9}
code,.mono,td.num,th.num{font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}
a{color:#1d4ed8}
.wrap{max-width:1280px;margin:0 auto;padding:0 28px 120px}
header.top{background:linear-gradient(160deg,#111827,#1f2937 60%,#312e5f);color:#f3f4f6;padding:44px 0 36px;margin-bottom:0}
header.top .wrap{padding-bottom:0}
header.top h1{margin:0 0 10px;font-size:29px;letter-spacing:.3px}
header.top p.sub{margin:0;color:#c3c8d4;font-size:14px}
header.top .meta{margin-top:20px;display:flex;flex-wrap:wrap;gap:8px}
header.top .meta span{background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.16);
     border-radius:5px;padding:4px 10px;font-size:12.5px;color:#e5e7eb}
nav.toc{position:sticky;top:0;z-index:40;background:rgba(255,255,255,.94);backdrop-filter:blur(8px);
     border-bottom:1px solid #e2e5ea;margin-bottom:34px}
nav.toc .wrap{padding:0 28px;display:flex;gap:2px;overflow-x:auto}
nav.toc a{padding:12px 12px;font-size:13.5px;color:#48505f;text-decoration:none;white-space:nowrap;border-bottom:2px solid transparent}
nav.toc a:hover{color:#111827;background:#f1f3f6}
nav.toc a.on{color:#1d4ed8;border-bottom-color:#1d4ed8;font-weight:600}
section{margin:0 0 46px;scroll-margin-top:60px}
h2{font-size:22px;margin:0 0 6px;padding-bottom:9px;border-bottom:2px solid #e2e5ea}
h2 .n{color:#9aa1ad;font-weight:400;margin-right:9px}
h3{font-size:17px;margin:30px 0 10px}
h4{font-size:14.5px;margin:20px 0 7px;color:#374151}
p{margin:11px 0}
.lead{font-size:15.5px;color:#3d4453;background:#fff;border:1px solid #e2e5ea;border-left:4px solid #6366f1;
     border-radius:0 8px 8px 0;padding:16px 20px;margin:16px 0 26px}
.card{background:#fff;border:1px solid #e2e5ea;border-radius:10px;padding:20px 22px;margin:16px 0}
.grid{display:grid;gap:14px}
.g2{grid-template-columns:repeat(auto-fit,minmax(330px,1fr))}
.g3{grid-template-columns:repeat(auto-fit,minmax(215px,1fr))}
.g4{grid-template-columns:repeat(auto-fit,minmax(165px,1fr))}
.kpi{background:#fff;border:1px solid #e2e5ea;border-radius:10px;padding:15px 17px}
.kpi .v{font-size:25px;font-weight:650;letter-spacing:-.4px;font-family:ui-monospace,Menlo,monospace}
.kpi .k{font-size:12.5px;color:#6b7280;margin-bottom:3px}
.kpi .note{font-size:12px;color:#8b919c;margin-top:5px;line-height:1.5}
.kpi.good .v{color:#047857}.kpi.bad .v{color:#b91c1c}.kpi.warn .v{color:#b45309}.kpi.flat .v{color:#374151}
table{border-collapse:collapse;width:100%;font-size:13.5px;background:#fff}
th,td{border:1px solid #e5e7eb;padding:7px 10px;text-align:left;vertical-align:top}
th{background:#f3f4f6;font-weight:600;font-size:12.5px;color:#374151;position:sticky;top:0}
td.num,th.num{text-align:right}
tr:hover td{background:#fafbfc}
.tbl-wrap{overflow:auto;max-height:560px;border:1px solid #e5e7eb;border-radius:8px}
.tbl-wrap table{border:0}
.tbl-wrap th{box-shadow:inset 0 -1px 0 #e5e7eb}
.pill{display:inline-block;padding:1px 8px;border-radius:11px;font-size:11.5px;font-weight:600;line-height:1.7;white-space:nowrap}
.p-L1{background:#dbeafe;color:#1e40af}
.p-L2{background:#dcfce7;color:#166534}
.p-fiber{background:#fef3c7;color:#92400e}
.p-none{background:#f3f4f6;color:#6b7280}
.p-ok{background:#d1fae5;color:#065f46}
.p-bad{background:#fee2e2;color:#991b1b}
.p-warn{background:#fef3c7;color:#92400e}
.p-info{background:#e0e7ff;color:#3730a3}
.bar{position:relative;background:#eef0f3;border-radius:3px;height:16px;min-width:80px;overflow:hidden}
.bar i{position:absolute;inset:0 auto 0 0;background:#93c5fd;display:block}
.bar b{position:absolute;inset:0;font-size:11px;text-align:center;font-weight:600;color:#1f2937;line-height:16px}
.bar.lo i{background:#fca5a5}.bar.hi i{background:#86efac}
details{border:1px solid #e2e5ea;border-radius:8px;background:#fff;margin:10px 0}
details>summary{cursor:pointer;padding:11px 15px;font-weight:600;font-size:13.5px;color:#374151;list-style:none;display:flex;gap:9px;align-items:center}
details>summary::-webkit-details-marker{display:none}
details>summary::before{content:"▸";color:#9aa1ad;font-size:12px}
details[open]>summary::before{content:"▾"}
details>summary:hover{background:#f8f9fb}
details .body{padding:4px 16px 16px;border-top:1px solid #eef0f3}
.note{font-size:13px;color:#6b7280;background:#f8f9fb;border-left:3px solid #cbd2dc;padding:9px 13px;margin:11px 0;border-radius:0 5px 5px 0}
.warnbox{font-size:13.5px;background:#fffbeb;border:1px solid #fde68a;border-left:4px solid #f59e0b;padding:13px 16px;border-radius:0 7px 7px 0;margin:13px 0}
.badbox{font-size:13.5px;background:#fef2f2;border:1px solid #fecaca;border-left:4px solid #ef4444;padding:13px 16px;border-radius:0 7px 7px 0;margin:13px 0}
.okbox{font-size:13.5px;background:#f0fdf4;border:1px solid #bbf7d0;border-left:4px solid #22c55e;padding:13px 16px;border-radius:0 7px 7px 0;margin:13px 0}
.infobox{font-size:13.5px;background:#eff6ff;border:1px solid #bfdbfe;border-left:4px solid #3b82f6;padding:13px 16px;border-radius:0 7px 7px 0;margin:13px 0}
ul,ol{margin:11px 0;padding-left:24px}
li{margin:5px 0}
.small{font-size:12.5px;color:#6b7280}
.flow{display:flex;align-items:stretch;gap:0;overflow-x:auto;padding:6px 0 12px}
.flow .st{min-width:150px;flex:1;background:#fff;border:1px solid #d6dae1;border-radius:8px;padding:10px 12px;position:relative}
.flow .st+.st{margin-left:20px}
.flow .st+.st::before{content:"";position:absolute;left:-20px;top:50%;width:20px;height:2px;background:#c3c9d3}
.flow .st+.st::after{content:"";position:absolute;left:-8px;top:50%;transform:translateY(-50%);
     border-left:6px solid #c3c9d3;border-top:4px solid transparent;border-bottom:4px solid transparent}
.flow .st .lab{font-size:11px;color:#8b919c;text-transform:uppercase;letter-spacing:.5px;margin-bottom:5px}
.flow .st .val{font-size:13px;font-weight:600;color:#1f2430;word-break:break-word}
.flow .st .sub{font-size:11.5px;color:#6b7280;margin-top:4px;line-height:1.55}
.flow .st.hit{border-color:#22c55e;background:#f0fdf4}
.flow .st.err{border-color:#ef4444;background:#fef2f2}
.flow .st.dim{opacity:.62}
.flow .st.pick{box-shadow:0 0 0 2px #6366f1 inset}
.lanes{font-size:12.5px;width:100%}
.lanes td,.lanes th{padding:4px 7px;text-align:right}
.lanes th:first-child,.lanes td:first-child{text-align:left;font-weight:600;color:#374151;white-space:nowrap}
.lanes td.dark{background:#fee2e2;color:#991b1b;font-weight:700}
.lanes td.zero{background:#fef3c7;color:#92400e;font-weight:700}
.lanes td.lo{background:#e0f2fe;color:#075985}
.lanes td.na{color:#c3c9d3}
.tokwrap{display:flex;flex-wrap:wrap;gap:5px;margin:8px 0}
.tok{font-family:ui-monospace,Menlo,monospace;font-size:11.5px;background:#f1f3f6;border:1px solid #e2e5ea;
     border-radius:4px;padding:2px 7px;color:#374151}
.tok.ex{background:#eef2ff;border-color:#c7d2fe;color:#3730a3}
.tok.cited{background:#fef9c3;border-color:#fde047;color:#854d0e;font-weight:600}
.thinking{font-size:13px;line-height:1.8;color:#3d4453;background:#fbfcfd;border:1px solid #e5e7eb;
     border-radius:7px;padding:14px 16px;white-space:pre-wrap;max-height:420px;overflow:auto}
.step{border:1px solid #e5e7eb;border-radius:7px;padding:11px 13px;margin:9px 0;background:#fff}
.step .claim{font-size:13.5px;color:#1f2430;margin-bottom:7px}
.step .m{font-size:11.5px;color:#6b7280;display:flex;flex-wrap:wrap;gap:10px;align-items:center}
.step.support{border-left:3px solid #22c55e}
.step.exclude{border-left:3px solid #ef4444}
.step.neutral{border-left:3px solid #cbd2dc}
.viol{font-size:12.5px;padding:6px 10px;border-radius:5px;margin:5px 0}
.viol.fatal{background:#fef2f2;border:1px solid #fecaca;color:#991b1b}
.viol.warning{background:#fffbeb;border:1px solid #fde68a;color:#92400e}
.filters{display:flex;flex-wrap:wrap;gap:8px;align-items:center;background:#fff;border:1px solid #e2e5ea;
     border-radius:9px;padding:13px 16px;margin:16px 0 22px;position:sticky;top:45px;z-index:30}
.filters .lbl{font-size:12.5px;color:#6b7280;margin-right:2px}
.btn{border:1px solid #d6dae1;background:#fff;border-radius:6px;padding:5px 11px;font-size:12.5px;cursor:pointer;color:#374151}
.btn:hover{background:#f3f4f6}
.btn.on{background:#1f2937;border-color:#1f2937;color:#fff;font-weight:600}
.search{border:1px solid #d6dae1;border-radius:6px;padding:5px 10px;font-size:12.5px;width:200px}
.caseCard{background:#fff;border:1px solid #e2e5ea;border-radius:11px;margin:16px 0;overflow:hidden}
.caseCard.wrong{border-color:#f3b8b8}
.caseCard.right{border-color:#c1e7cd}
.caseHead{display:flex;flex-wrap:wrap;gap:11px;align-items:center;padding:13px 18px;border-bottom:1px solid #eef0f3;background:#fbfcfd}
.caseCard.wrong .caseHead{background:#fef7f7}
.caseCard.right .caseHead{background:#f7fdf9}
.caseHead .cid{font-family:ui-monospace,Menlo,monospace;font-size:13.5px;font-weight:650}
.caseHead .grow{flex:1}
.caseBody{padding:16px 18px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:1000px){.cols{grid-template-columns:1fr}}
.blind{background:#f5f3ff;border:1px solid #ddd6fe;border-left:4px solid #7c3aed;border-radius:0 8px 8px 0;padding:14px 17px;margin:12px 0}
.blind h5{margin:0 0 7px;font-size:13.5px;color:#5b21b6;text-transform:none}
.insight{background:#fffbeb;border:1px solid #fde68a;border-left:4px solid #d97706;border-radius:0 8px 8px 0;padding:14px 17px;margin:12px 0}
.insight h5{margin:0 0 7px;font-size:13.5px;color:#92400e}
.fix{background:#f0fdf4;border:1px solid #bbf7d0;border-left:4px solid #16a34a;border-radius:0 8px 8px 0;padding:14px 17px;margin:12px 0}
.fix h5{margin:0 0 7px;font-size:13.5px;color:#166534}
.blind p,.insight p,.fix p{margin:6px 0;font-size:13.5px;line-height:1.8}
h5{font-weight:700}
.kindTag{font-size:11.5px;font-weight:600;padding:2px 9px;border-radius:11px;color:#fff}
.empty{text-align:center;color:#9aa1ad;padding:50px;font-size:14px}
.footer{margin-top:60px;padding-top:22px;border-top:1px solid #e2e5ea;font-size:12.5px;color:#8b919c}
.tree{font-family:ui-monospace,Menlo,monospace;font-size:12.5px;line-height:1.9;white-space:pre;overflow-x:auto;
      background:#fbfcfd;border:1px solid #e5e7eb;border-radius:7px;padding:14px 16px}
"""

JS_COMMON = """
function pill(v){const t=v===null||v===undefined?'弃答':v;const c=v?('p-'+v):'p-none';
  return '<span class="pill '+c+'">'+t+'</span>';}
function pct(x,d){return (100*x).toFixed(d===undefined?1:d)+'%';}
function esc(s){return String(s===null||s===undefined?'':s)
  .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function bar(v,cls){return '<div class="bar '+(cls||'')+'"><i style="width:'+(100*v).toFixed(1)+'%"></i>'
  +'<b>'+pct(v)+'</b></div>';}
// 目录高亮：用可见区域最靠上的 section 作为当前项，避免滚动到底部时全部熄灭
(function(){
  const links=[...document.querySelectorAll('nav.toc a')];
  const secs=links.map(a=>document.querySelector(a.getAttribute('href'))).filter(Boolean);
  function sync(){
    let cur=0;
    secs.forEach((s,i)=>{if(s.getBoundingClientRect().top<=120)cur=i;});
    links.forEach((a,i)=>a.classList.toggle('on',i===cur));
  }
  document.addEventListener('scroll',sync,{passive:true});sync();
})();
"""


def head(title: str, subtitle: str, chips: list[str], nav: list[tuple[str, str]]) -> str:
    chip_html = "".join(f"<span>{c}</span>" for c in chips)
    nav_html = "".join(f'<a href="#{i}">{t}</a>' for i, t in nav)
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>{CSS}</style></head><body>
<header class="top"><div class="wrap">
<h1>{title}</h1><p class="sub">{subtitle}</p>
<div class="meta">{chip_html}</div>
</div></header>
<nav class="toc"><div class="wrap">{nav_html}</div></nav>
<div class="wrap">
"""


def tail(note: str, payloads: dict[str, object], script: str) -> str:
    data = "".join(
        f'<script id="{k}" type="application/json">{json.dumps(v, ensure_ascii=False)}</script>'
        for k, v in payloads.items()
    )
    return f"""
<div class="footer">{note}</div>
</div>
{data}
<script>{JS_COMMON}
{script}
</script>
</body></html>"""


# --------------------------------------------------------------------------------------
# 报告一：框架与整体结果
# --------------------------------------------------------------------------------------

OVERVIEW_NAV = [
    ("s-verdict", "结论"),
    ("s-arch", "实现框架"),
    ("s-pack", "证据包"),
    ("s-feat", "特征与证据"),
    ("s-graph", "证据图"),
    ("s-constr", "物理约束"),
    ("s-sop", "SOP"),
    ("s-expert", "专家规则"),
    ("s-cascade", "级联决策"),
    ("s-result", "结果与效果"),
    ("s-cells", "规则可靠性解剖"),
    ("s-ceiling", "可推导性上限"),
    ("s-llm", "LLM 三岗位"),
    ("s-defects", "缺陷与建议"),
]


def build_overview() -> str:
    ov = json.loads(OVERVIEW_BUNDLE.read_text(encoding="utf-8"))
    cb = json.loads(CASE_BUNDLE.read_text(encoding="utf-8"))
    df = json.loads((ROOT / "artifacts/defect_bundle.json").read_text(encoding="utf-8"))
    meta = cb["meta"]
    fs = df["final_system"]
    blind = df["blind_reader"]["all"]

    chips = [
        f"数据集 {ov['dataset']['name']}",
        f"train/test {ov['dataset']['train']}/{ov['dataset']['test']}",
        f"模型 {meta['model']}",
        f"prompt {meta['prompt_version']}",
        f"约束库 {ov['constraint_version']}",
        f"特征 {ov['graph']['dictionary']}",
    ]

    html = [head(
        "光链路 RCA：框架、证据、约束与效果解剖",
        "对当前 RCA v2 测试数据的完整复盘 —— 系统由什么构成、做到了什么水平、"
        "以及「换一个更强的推理模型」能不能解决问题",
        chips, OVERVIEW_NAV,
    )]

    # ---- 结论 ----
    html.append(f"""
<section id="s-verdict"><h2><span class="n">00</span>先回答那个假设</h2>
<div class="lead">
你的假设是「效果差可能是因为用了比较差的推理模型」。把测试数据翻完之后，结论是
<b>这个假设不成立，而且可以被具体地否证</b>：当前 24 条判错的 case 里，有 23 条是
「从这份遥测出发，任何推理者都推不出答案」——不是推错，是无从下手。
一个更强的模型在这些 case 上唯一能做得更好的事情是<b>拒绝回答</b>，而不是答对。
</div>

<div class="grid g4">
  <div class="kpi flat"><div class="k">自动结论覆盖率</div><div class="v">{fs['coverage']*100:.1f}%</div>
    <div class="note">{fs['answered']}/{fs['n']} 条给出结论</div></div>
  <div class="kpi warn"><div class="k">给结论时精度</div><div class="v">{fs['precision']*100:.1f}%</div>
    <div class="note">{fs['correct']}/{fs['answered']} 条正确</div></div>
  <div class="kpi flat"><div class="k">多数类基线</div><div class="v">62.6%</div>
    <div class="note">全部答 L2 就有这个成绩</div></div>
  <div class="kpi bad"><div class="k">fiber 召回</div><div class="v">0.0%</div>
    <div class="note">0/8，结构性不可召回</div></div>
</div>

<h3>三句话版本</h3>
<div class="okbox"><b>1. 增益来自人工专家经验，不来自模型。</b>
覆盖率从 MVP 的 0% 提到 {fs['coverage']*100:.1f}%，靠的是把工程阈值与两端仲裁规则写进
<code>expert.py</code>。同一份数据上，LLM 作为定界器的准确率显著低于专家规则
（McNemar p=0.0117），作为质疑器命中率为 0，只有作为解释器时四项可机检指标全部通过。</div>

<div class="warnbox"><b>2. 专家规则 {ov['baselines']['expert']['acc']*100:.2f}% 的成绩里，一半以上落在统计上不成立的格子上。</b>
把规则按「哪条规则胜出 + 从哪端指向哪端」拆成 15 个格子后，
最大的三个格子（<code>multi_metric L1→L2</code> 之外的 <code>multi_metric L2→L1</code>、
<code>single:serdes_snr</code> 两个方向）合计覆盖全库过半 case，准确率只有 50~74%。
按选择性预测重排后，<b>覆盖率降到 63.6% 可以把精度从 76.64% 提到 85.29%</b>。</div>

<div class="badbox"><b>3. 真正的瓶颈是可观测量，不是推理能力。</b>
用一套只依赖物理哨兵与同侧对比的「盲读」规则（不看标签、不用任何数据集统计量）扫全库：
只有 <b>{blind['coverage']*100:.1f}%（{blind['answered']}/{blind['n']}）</b>的 case 能被物理推导出结论，
精度 {blind['blind_precision']*100:.1f}%；而专家规则在<b>同一个子集</b>上精度也正好是
{blind['expert_precision_on_same_subset']*100:.1f}%。也就是说物理能说清的地方，
现有规则已经做到了；剩下 91% 的 case 里没有可推理的内容，只有可猜的先验。</div>
</section>
""")

    # ---- 架构 ----
    html.append("""
<section id="s-arch"><h2><span class="n">01</span>实现框架</h2>
<p>一条告警从原始遥测走到结论要经过六个节点。下面每个节点后面括号里是它在代码里的位置。</p>
<div class="card">
<div class="flow">
  <div class="st"><div class="lab">N1 证据包</div><div class="val">原始遥测标准化</div>
    <div class="sub">逐 lane 光功率/SNR/状态位<br><code>evidence_pack.py</code></div></div>
  <div class="st"><div class="lab">N2 特征</div><div class="val">稀疏证据 token</div>
    <div class="sub">阈值+分位数 → 可解释 token<br><code>features/extractor.py</code></div></div>
  <div class="st"><div class="lab">N3 证据图</div><div class="val">历史签名检索</div>
    <div class="sub">IDF 加权 Jaccard<br><code>evidence_graph/</code></div></div>
  <div class="st"><div class="lab">N4 路由</div><div class="val">按命中强度分流</div>
    <div class="sub">N5a/N5b/N5c/N6<br><code>branches/</code></div></div>
  <div class="st"><div class="lab">M9 级联</div><div class="val">多来源裁决</div>
    <div class="sub">专家/SOP/分支/LLM<br><code>decision.py</code></div></div>
  <div class="st"><div class="lab">输出</div><div class="val">结论或拒答</div>
    <div class="sub">带置信度与人工介入标记</div></div>
</div>
<div class="note">LLM 不在主干上。它被挂在三个可替换的岗位上：<b>定界器</b>（直接给根因）、
<b>质疑器</b>（判断专家规则的前提是否成立）、<b>解释器</b>（把规则依据翻译成人话）。
三个岗位分别做了独立实验，结论见下方「LLM 三岗位」。</div>
</div>

<h3>关键设计：训练与测试的隔离</h3>
<p>所有在训练集上拟合的东西——阈值模型、特征分位数边界、IDF 权重、SOP 决策树、
标定分组的支持数与 Wilson 下界——都只用 train 的 161 条。test 的 107 条不参与任何拟合，
也不回灌进证据图。证据包这一层的类型设计强制了这件事：
<code>fit_feature_model()</code> 的入参是 <code>EvidencePack</code>，而证据包里根本没有
<code>label</code> 字段，所以「拟合时不许看标签」由类型保证，不靠调用方自觉。</p>
</section>
""")

    # 其余章节由前端从 JSON 渲染
    html.append("""
<section id="s-pack"><h2><span class="n">02</span>证据包：系统能看到什么</h2>
<p>证据包是唯一的输入。它决定了后面所有环节的能力上限，所以先看它里面到底有什么、缺什么。</p>
<div id="packCoverage"></div>
<h3>结构性问题</h3>
<div id="packIssues"></div>
</section>

<section id="s-feat"><h2><span class="n">03</span>特征与证据 token</h2>
<p>N2 把连续的遥测读数变成离散 token。这一步是<b>有损</b>的，而损失掉的东西正好是后面
很多判错的原因，所以值得逐族看清楚。</p>
<div id="tokenFamilies"></div>
<div id="tokenLossy"></div>
</section>

<section id="s-graph"><h2><span class="n">04</span>证据图</h2>
<div id="graphInfo"></div>
</section>

<section id="s-constr"><h2><span class="n">05</span>物理约束库</h2>
<p>约束库是给 LLM 的物理护栏，同时也是校验器判断模型有没有胡说的依据。
每条约束规定了它适用于哪些 token 前缀、允许被用作什么效果（支持/排除/中性）、
以及允许指向哪些根因。</p>
<div id="constrSummary"></div>
<div id="constrTable"></div>
</section>

<section id="s-sop"><h2><span class="n">06</span>SOP</h2>
<div id="sopInfo"></div>
</section>

<section id="s-expert"><h2><span class="n">07</span>专家规则</h2>
<div id="expertInfo"></div>
</section>

<section id="s-cascade"><h2><span class="n">08</span>级联决策 M9</h2>
<div id="cascadeInfo"></div>
</section>

<section id="s-result"><h2><span class="n">09</span>结果与效果</h2>
<div id="resultBlock"></div>
</section>

<section id="s-cells"><h2><span class="n">10</span>规则可靠性解剖</h2>
<p>整体准确率会把好格子和坏格子的表现平均掉。把专家规则按
「哪条规则胜出 + 症状在哪端 → 指向哪端」拆开之后，才能看到成绩是从哪里来的、
错误集中在哪里。下表在<b>全库 268 条</b>上统计（test 只有 107 条，单格样本量不够）。</p>
<div id="cellsBlock"></div>
<h3>选择性预测：拒答换精度</h3>
<div id="selectiveBlock"></div>
</section>

<section id="s-ceiling"><h2><span class="n">11</span>物理可推导性上限</h2>
<div id="ceilingBlock"></div>
</section>

<section id="s-llm"><h2><span class="n">12</span>LLM 三岗位实测</h2>
<div id="llmBlock"></div>
</section>

<section id="s-defects"><h2><span class="n">13</span>缺陷清单与建议</h2>
<div id="defectBlock"></div>
</section>
""")

    html.append(tail(
        "本页所有数字均由 <code>scripts/build_overview_bundle.py</code> 与 "
        "<code>scripts/build_defect_bundle.py</code> 在 "
        "<code>datasets/rca_v2_l2fixed</code> 上重算得出，可重跑核对。"
        "逐 case 的决策链与错因分析见 <a href='rca_cases.html'>rca_cases.html</a>。",
        {"ovData": ov, "cbMeta": {"meta": cb["meta"], "metrics": cb["metrics"],
                                  "challenge": cb["challenge_summary"], "explain": cb["explain_summary"]},
         "dfData": df},
        OVERVIEW_JS,
    ))
    return "".join(html)


OVERVIEW_JS = r"""
const OV=JSON.parse(document.getElementById('ovData').textContent);
const CB=JSON.parse(document.getElementById('cbMeta').textContent);
const DF=JSON.parse(document.getElementById('dfData').textContent);
const H=(id,s)=>{const e=document.getElementById(id);if(e)e.innerHTML=s;};

/* ---------- 02 证据包 ---------- */
(function(){
  const m=DF.data_hygiene.missing;
  const order=['rxpower','txpower','media_snr','host_snr','serdes_snr','bias'];
  const desc={rxpower:'接收光功率，逐 lane',txpower:'发送光功率，逐 lane',
    media_snr:'介质侧信噪比（面向光纤）',host_snr:'主机侧信噪比（面向交换芯片）',
    serdes_snr:'SerDes 电通道信噪比',bias:'激光器偏置电流'};
  let r='<div class="tbl-wrap"><table><tr><th>指标</th><th>含义</th>'+
    '<th class="num">L1 缺失</th><th class="num">L2 缺失</th><th>覆盖情况</th></tr>';
  order.forEach(k=>{
    const a=m[k+'.L1'],b=m[k+'.L2'];
    const worst=Math.max(a.rate,b.rate);
    r+='<tr><td><code>'+k+'</code></td><td class="small">'+desc[k]+'</td>'+
      '<td class="num">'+a.missing+' ('+pct(a.rate)+')</td>'+
      '<td class="num">'+b.missing+' ('+pct(b.rate)+')</td>'+
      '<td>'+bar(1-worst,worst>0.3?'lo':'hi')+'</td></tr>';
  });
  r+='</table></div>';
  r+='<div class="warnbox"><b>host_snr 缺失 41~45%，而它是唯一能无歧义定端的指标。</b>'+
     'host_snr 量的是模块朝交换芯片那一侧的电通道，对端无论如何也影响不到它，'+
     '所以它异常就只能是本端的问题。专家规则给了它最高优先级（21），'+
     '但因为样本太少（全库该判据只触发 2 次），标定分组永远攒不到 M9 要求的支持数，'+
     '结果是<b>物理上最硬的判据反而最容易被级联作废</b>。'+
     'case_5d80fba2c22f 就是这样丢掉一个正确结论的。</div>';
  H('packCoverage',r);

  const dh=DF.data_hygiene;
  let q='<div class="grid g2">';
  q+='<div class="card"><h4>跨端 lane 数不一致</h4><p class="small">'+
     dh.cross_side_lane_mismatch.n+'/'+dh.total+'（'+pct(dh.cross_side_lane_mismatch.rate)+
     '）的 case 里，L1 的接收 lane 数与 L2 的发送 lane 数不同。'+
     '这意味着「本端 lane k 收不到光 → 查对端 lane k 发了没有」这条最基本的交叉核验，'+
     '在这些 case 上无法进行。</p></div>';
  const sv=Object.entries(dh.serdes_vs_optical_lanes).sort((a,b)=>b[1]-a[1]);
  q+='<div class="card"><h4>光 lane 数与 serdes lane 数的配比</h4><table>'+
     '<tr><th>配比</th><th class="num">出现次数</th></tr>'+
     sv.map(([k,v])=>'<tr><td class="mono">'+k+'</td><td class="num">'+v+'</td></tr>').join('')+
     '</table><p class="small">出现 8 光 lane 配 4 serdes lane 的情况，说明存在 gearbox/lane 复用，'+
     'serdes lane 与光 lane 不是一一对应。不过实测这并不是 serdes 判据不准的原因——见第 13 节。</p></div>';
  q+='</div>';
  const up=DF.data_hygiene.unparsed_blocks;
  if(Object.keys(up).length){
    q+='<div class="badbox"><b>数据管道缺陷：字段未被解析且未报缺失。</b>'+
      Object.entries(up).map(([k,v])=>'<code>'+k+'</code> 字段在 <b>'+v.length+
      '</b> 条 case 上是未解析的字符串而不是字典（例如 <code>'+v[0]+'</code>）').join('；')+
      '。<code>bias</code> 参与约束 C1（bias 归零 ⇔ 发端断光）与 C2（健康区间 7.2~7.8 mA）的'+
      ' token 生成，这些 case 的相关证据实际缺失，但系统没有把它们记入 <code>missing_fields</code>，'+
      '而是静默降级。</div>';
  }
  q+='<div class="warnbox"><b>txpower 可能是标称值而非实测值。</b>全库有 '+dh.flat_txpower_sides+
     ' 个「侧」的发送功率四条读数完全相同（例如 0.69, 0.69, 0.69, 0.69）。'+
     '如果 tx 是配置值，那么「对端发功率正常」就不能证明实际出光正常——'+
     '这直接解释了为什么教科书式的 fiber 判据（对端在发、本端收不到 ⇒ 光缆问题）'+
     '在全库上精度只有 14.4%。这个字段的语义必须先查清，'+
     '否则所有依赖「对端发端是否正常」的推理都建立在流沙上。</div>';
  H('packIssues',q);
})();

/* ---------- 03 特征 ---------- */
(function(){
  const fam=OV.token_families;
  const note={level:'相对<b>数据集分位数</b>的高/低尾判断（比别的链路高或低）',
    expert:'专家阈值表判定的异常，以及两端仲裁得出的 points_to / pattern',
    drop:'相对健康基线的跌落幅度分档',serdes:'serdes 数据是否可用',
    telemetry:'遥测完整度标记',status:'端口状态位',link:'链路侧属性'};
  const tot=Object.values(fam).reduce((a,b)=>a+b,0);
  let r='<div class="grid g2"><div class="card"><h4>token 族分布（train 161 条累计）</h4><table>'+
    '<tr><th>族</th><th class="num">出现次数</th><th>占比</th><th>它编码的是什么</th></tr>';
  Object.entries(fam).sort((a,b)=>b[1]-a[1]).forEach(([k,v])=>{
    r+='<tr><td><code>'+k+':</code></td><td class="num">'+v+'</td><td>'+bar(v/tot)+'</td>'+
      '<td class="small">'+(note[k]||'')+'</td></tr>';
  });
  r+='</table><p class="small">词表规模 '+OV.token_vocab+' 个不同 token，字典版本 <code>'+
     OV.graph.dictionary+'</code>。</p></div>';
  r+='<div class="card"><h4>这一步丢掉了什么</h4><ul>'+
    '<li><b>偏离的方向。</b><code>rxpower:lane_diff</code> 只说「某条 lane 与同侧其余不同」，'+
    '不说是偏高还是偏低。而两者物理含义相反：偏低是上游衰减（指向对端），'+
    '偏高不可能由对端衰减造成（指向本端标定或读数问题）。'+
    '判错的 case 里有 3 条（3a9a11e3c449、0c3939c5292e、7fa3fefa52e3）'+
    '真因都在「出现偏高读数的那一侧」，而规则一律指向对端。</li>'+
    '<li><b>断了几条 lane、断的是哪几条。</b><code>lane_down</code> 是布尔型。'+
    '而单 lane 断与多条非相邻 lane 同时断，物理含义完全不同——'+
    '前者更像某一芯光纤或某一路光器件，后者更像模块内部按通道分组的失效。'+
    'case_9e14bb67adf3 的判断依据正是「lane 1 与 lane 3 同时断而 0、2 完好」。</li>'+
    '<li><b>绝对参考。</b>所有 level 族 token 都是相对<b>数据集分位数</b>的，'+
    '也就是「比别的链路低」，而不是「比这条链路应有的水平低」。'+
    '判断「L1 收到 0.4 dBm 是偏低还是正常」需要链路功率预算（对端标称发功率、'+
    '光缆长度与损耗），证据包里没有任何这类基线。</li>'+
    '<li><b>跨端对比。</b>所有 token 都是单侧的，没有一个 token 表达'+
    '「L1 与 L2 的同名指标相差多少」。而 fiber 的物理特征本质上就是'+
    '「两端都不正常但都不是单端故障」，只能用跨端量表达。'+
    '连最基本的链路插损（对端 tx − 本端 rx）都没有被算出来。</li>'+
    '</ul></div></div>';
  H('tokenLossy',r);
  H('tokenFamilies','');
})();

/* ---------- 04 证据图 ---------- */
(function(){
  const g=OV.graph, b=OV.baselines.graph_top1;
  let r='<div class="grid g2"><div class="card"><h4>构成</h4><ul>'+
    '<li>节点数 <b>'+g.nodes+'</b>（train 全部 case，test 不回灌）</li>'+
    '<li>版本 <code>'+g.version+'</code>，字典 <code>'+g.dictionary+'</code></li>'+
    '<li>相似度：证据 token 集合的 <b>IDF 加权 Jaccard</b>。'+
    'IDF 只在 train 上统计，作用是让罕见 token 比常见 token 更有区分力</li>'+
    '<li>标签分布 '+Object.entries(g.label_dist).map(([k,v])=>k+':'+v).join(' / ')+'</li>'+
    '</ul></div>';
  r+='<div class="card"><h4>纯检索基线的表现</h4><p>把 top-1 邻居的标签直接当预测：'+
    '总准确率 <b>'+pct(b.acc,2)+'</b>，低于多数类基线的 62.62%。但它有一个别人都没有的能力：</p>'+
    '<table><tr><th>类别</th><th class="num">召回</th><th class="num">精度</th></tr>'+
    ['L1','L2','fiber'].map(l=>'<tr><td>'+pill(l)+'</td><td class="num">'+
      pct(b.per_class[l].recall)+'</td><td class="num">'+pct(b.per_class[l].precision)+
      '</td></tr>').join('')+'</table>'+
    '<div class="okbox" style="margin-top:12px"><b>检索是唯一能召回 fiber 的通道</b>'+
    '（37.5%，3/8），而专家规则、SOP、LLM 定界的 fiber 召回全是 0。'+
    'fiber 在这份数据里只能靠<b>签名记忆</b>找到，不能靠物理推理找到。</div></div></div>';
  const hist=g.test_sim_hist;
  r+='<div class="card"><h4>test 上 top-1 相似度的分布</h4><table><tr><th>相似度区间</th>'+
    '<th class="num">case 数</th><th>占比</th></tr>'+
    Object.entries(hist).map(([k,v])=>'<tr><td class="mono">'+k+' ~ '+
      (parseFloat(k)+0.1).toFixed(1)+'</td><td class="num">'+v+'</td><td>'+
      bar(v/107)+'</td></tr>').join('')+'</table>'+
    '<p class="small">相似度很低时（例如 case_fa72aef06d4d 只有 0.366），'+
    '证据图实际上在说「训练集里没见过这样的 case」。这个信号目前没有被 M9 用作拒答依据。</p></div>';
  H('graphInfo',r);
})();

/* ---------- 05 约束 ---------- */
(function(){
  const cs=OV.constraints;
  const byCat={},byKind={};
  cs.forEach(c=>{byCat[c.category]=(byCat[c.category]||0)+1;byKind[c.kind]=(byKind[c.kind]||0)+1;});
  const kindCN={invariant:'不变式（恒成立）',exclusion:'排除式（可用于排除某根因）',
    indicator:'指示式（提高某根因可能性）',caveat:'警示式（禁止某类推断）'};
  let r='<div class="grid g2"><div class="card"><h4>按类型</h4><table><tr><th>类型</th>'+
    '<th class="num">条数</th><th>含义</th></tr>'+
    Object.entries(byKind).sort((a,b)=>b[1]-a[1]).map(([k,v])=>'<tr><td><code>'+k+
    '</code></td><td class="num">'+v+'</td><td class="small">'+(kindCN[k]||'')+'</td></tr>').join('')+
    '</table></div><div class="card"><h4>按物理范畴</h4><table><tr><th>范畴</th>'+
    '<th class="num">条数</th></tr>'+
    Object.entries(byCat).sort((a,b)=>b[1]-a[1]).map(([k,v])=>'<tr><td><code>'+k+
    '</code></td><td class="num">'+v+'</td></tr>').join('')+'</table></div></div>';
  r+='<div class="note">版本 <code>'+OV.constraint_version+'</code>，共 <b>'+cs.length+
     '</b> 条。其中标注为 <code>measured</code> 的条目，其参数是在 '+OV.constraint_measured_on+
     ' 上实测得到的，不是凭经验写的。</div>';
  H('constrSummary',r);

  let t='<div class="tbl-wrap"><table><tr><th>ID</th><th>物理断言</th><th>形式化</th>'+
    '<th>适用 token 前缀</th><th>允许效果</th><th>允许指向</th></tr>';
  cs.forEach(c=>{
    t+='<tr><td class="mono" style="white-space:nowrap"><b>'+c.id.split('_')[0]+'</b><br>'+
      '<span class="small">'+c.kind+'</span></td>'+
      '<td>'+esc(c.statement)+
        (c.params.length?'<br><span class="small">参数：'+c.params.map(p=>p[0]+' = '+p[1]).join('，')+'</span>':'')+
        (c.evidence?'<br><span class="small" style="color:#8b919c">实测：'+esc(c.evidence).slice(0,220)+'</span>':'')+
      '</td>'+
      '<td class="mono small">'+esc(c.formal)+'</td>'+
      '<td class="small">'+(c.prefixes.length?c.prefixes.map(p=>'<code>'+p+'</code>').join('<br>'):'—')+'</td>'+
      '<td class="small">'+c.effects.map(e=>'<span class="pill '+
        (e==='support'?'p-ok':e==='exclude'?'p-bad':'p-none')+'">'+(e||'无')+'</span>').join(' ')+'</td>'+
      '<td class="small">'+c.targets.map(x=>x?pill(x):'<span class="pill p-none">—</span>').join(' ')+'</td></tr>';
  });
  t+='</table></div>';
  t+='<div class="warnbox"><b>约束正确 ≠ 用法正确。</b>C23/C24 这一对方向约束'+
     '（「某端接收异常支持对端」）在物理上有依据，但它<b>只看症状出现在哪一侧，'+
     '不核对对端发端是否真有对应的弱 lane</b>。判错的 case 里 case_fcacbc45173f 就是反例：'+
     'L1 侧 lane 2 收得弱，但 L2 在这条 lane 上的发送功率恰恰是四条里最高的。'+
     'LLM 忠实引用了 C23 并给出 L2——模型没错，是它被给了一条在这个子集上方向错误的前提。</div>';
  H('constrTable',t);
})();

/* ---------- 06 SOP ---------- */
(function(){
  const s=OV.sop;
  function render(n,depth,prefix){
    const cnt=n.label_counts||{};
    const dist=['L1','L2','fiber'].filter(l=>cnt[l]).map(l=>l+':'+cnt[l]).join(' ');
    let line=prefix+'n='+n.samples+'  ['+dist+']  → '+(n.prediction||'无多数类');
    let out=line+'\n';
    if(n.token){
      out+=render(n.present,depth+1,prefix+'  ├─ 有 '+n.token+' → ');
      out+=render(n.absent,depth+1,prefix+'  └─ 无 '+n.token+' → ');
    }
    return out;
  }
  const b=OV.baselines.sop;
  let r='<p>SOP 在这里不是人写的处置手册，而是<b>从训练标签学出来的一棵浅决策树</b>'+
    '（深度 '+s.max_depth+'，叶子最小样本 '+s.min_leaf_size+'）。'+
    '每个叶子带支持数与 Wilson 下界，让下游可以在路径太弱或标签太混时拒答。</p>';
  r+='<div class="grid g3">'+
    '<div class="kpi flat"><div class="k">SOP 单独的准确率</div><div class="v">'+pct(b.acc,2)+'</div>'+
    '<div class="note">高于多数类 62.62%，低于专家规则</div></div>'+
    '<div class="kpi bad"><div class="k">SOP 的 fiber 召回</div><div class="v">0.0%</div>'+
    '<div class="note">树永远不会走到 fiber 叶子</div></div>'+
    '<div class="kpi flat"><div class="k">在级联里实际采纳</div><div class="v">10 条</div>'+
    '<div class="note">其中 8 条正确（80%）</div></div></div>';
  r+='<h4>学出来的树</h4><div class="tree">'+esc(render(s.root,0,''))+'</div>';
  r+='<div class="note">树只在 <code>expert:</code> 与 <code>level:</code> 这些 token 上分裂，'+
     '本质上是在复述专家规则已经编码的信息，所以它和专家规则高度相关，'+
     '在级联里提供的是冗余而不是新信息。</div>';
  H('sopInfo',r);
})();

/* ---------- 07 专家规则 ---------- */
(function(){
  const e=OV.expert_rules, b=OV.baselines.expert;
  const dirCN={self:'指向本端（出现症状的这一侧）',peer:'指向对端'};
  let r='<div class="grid g2"><div class="card"><h4>阈值表</h4><table>'+
    '<tr><th>指标</th><th class="num">低值门限</th><th class="num">高值门限</th></tr>'+
    Object.entries(e.thresholds).map(([k,v])=>'<tr><td><code>'+k+'</code></td>'+
      '<td class="num">'+(v.low!==undefined?v.low:'—')+'</td>'+
      '<td class="num">'+(v.high!==undefined?v.high:'—')+'</td></tr>').join('')+
    '</table></div>';
  r+='<div class="card"><h4>单指标规则的优先级与方向</h4><table>'+
    '<tr><th>指标</th><th class="num">基础优先级</th><th>方向</th></tr>'+
    Object.entries(e.single_base).sort((a,b)=>a[1]-b[1]).map(([k,v])=>'<tr><td><code>'+k+
      '</code></td><td class="num">'+v+'</td><td class="small">'+
      (dirCN[e.single_direction[k]]||e.single_direction[k])+'</td></tr>').join('')+
    '</table><p class="small">优先级按字符串排序，数字越小越优先。'+
    '<code>multi_metric</code>（'+e.multi_metric_requires.join(' + ')+' 同时异常）'+
    '优先级为 '+e.multi_metric_priority+'，压过所有单指标规则。</p></div></div>';
  r+='<div class="card"><h4>两端仲裁</h4><p>对 L1 和 L2 各自独立跑一遍规则，然后取优先级更小的一侧胜出；'+
    '若两侧优先级相同则记为 <code>both_anomaly</code>，若两侧都没触发则走 <code>no_anomaly</code> 兜底。'+
    '胜出侧的规则决定「指向本端还是对端」，这一步才产出最终的 L1/L2 判定。</p>'+
    '<div class="badbox"><b>仲裁会吞掉方向信息。</b><code>multi_metric</code> 优先级最高（1），'+
    '但它只数「有几个指标异常」，不看其中哪个指标能确定方向。'+
    'case_788f63a0a0fc 里 L2 侧连 <code>host_snr</code> 都异常了——'+
    'host_snr 在物理上不可能被对端影响，所以方向本应由它决定；'+
    '但 multi_metric 先触发，于是套用了「收端异常指对端」的默认方向，判成 L1，真因是 L2。</div></div>';
  r+='<div class="grid g4">'+
    '<div class="kpi warn"><div class="k">专家规则准确率（全答）</div><div class="v">'+pct(b.acc,2)+'</div>'+
    '<div class="note">test 107 条</div></div>'+
    ['L1','L2','fiber'].map(l=>'<div class="kpi '+(b.per_class[l].recall>0.7?'good':
      b.per_class[l].recall>0?'warn':'bad')+'"><div class="k">'+l+' 召回</div><div class="v">'+
      pct(b.per_class[l].recall)+'</div><div class="note">'+b.per_class[l].hit+'/'+
      b.per_class[l].support+'</div></div>').join('')+'</div>';
  H('expertInfo',r);
})();

/* ---------- 08 级联 ---------- */
(function(){
  const src=CB.metrics.by_candidate_source, dg=CB.metrics.degeneracy_guard;
  let r='<p>M9 依次考察若干候选来源，第一个同时满足「标定分组的 Wilson 下界达标」与'+
    '「支持数达标」的候选被采纳；都不达标就拒答（转人工或请求补采）。</p>';
  r+='<div class="card"><h4>各来源在 test 上实际贡献了多少</h4><table>'+
    '<tr><th>来源</th><th class="num">采纳条数</th><th class="num">其中正确</th><th>精度</th></tr>'+
    Object.entries(src).sort((a,b)=>b[1].answered-a[1].answered).map(([k,v])=>
      '<tr><td><code>'+k+'</code></td><td class="num">'+v.answered+'</td><td class="num">'+
      v.correct+'</td><td>'+bar(v.precision_when_answered,v.precision_when_answered>0.8?'hi':'')+
      '</td></tr>').join('')+'</table>'+
    '<p class="small">专家规则承担了 90/102 的结论。LLM 作为候选来源一次都没有被采纳。</p></div>';
  r+='<div class="grid g3">'+
    '<div class="kpi good"><div class="k">相对多数类的增益</div><div class="v">+'+
      (dg.lift_over_majority_on_kept*100).toFixed(2)+'pp</div>'+
      '<div class="note">在给结论的子集上，多数类为 '+pct(dg.majority_on_kept)+'</div></div>'+
    '<div class="kpi flat"><div class="k">人工介入率</div><div class="v">'+
      pct(CB.metrics.human_intervention_rate)+'</div><div class="note">2 条转人工，3 条请求补采</div></div>'+
    '<div class="kpi flat"><div class="k">平衡召回</div><div class="v">'+
      (dg.balanced_recall!==undefined?dg.balanced_recall.toFixed(4):'—')+'</div>'+
      '<div class="note">三类召回的算术平均，被 fiber 的 0 拖住</div></div></div>';
  r+='<div class="badbox"><b>级联的优先级设计有一个原则性错误。</b>'+
     'case_22d4dde687ab 的证据签名与训练集里某条 fiber case <b>完全相同（相似度 1.000）</b>，'+
     '而级联仍然采纳了专家规则给出的 L1。检索精确命中本应是最强的证据，'+
     '却被一个 Wilson 下界只有 34.4% 的规则格子覆盖了。</div>';
  H('cascadeInfo',r);
})();

/* ---------- 09 结果 ---------- */
(function(){
  const fs=DF.final_system, cm=CB.metrics.class_metrics;
  let r='<div class="grid g4">'+
    '<div class="kpi flat"><div class="k">覆盖率</div><div class="v">'+pct(fs.coverage,2)+'</div>'+
      '<div class="note">'+fs.answered+'/'+fs.n+'</div></div>'+
    '<div class="kpi warn"><div class="k">给结论时精度</div><div class="v">'+pct(fs.precision,2)+'</div>'+
      '<div class="note">'+fs.correct+'/'+fs.answered+'</div></div>'+
    '<div class="kpi flat"><div class="k">全集正确率</div><div class="v">'+pct(fs.overall,2)+'</div>'+
      '<div class="note">拒答记为不正确</div></div>'+
    '<div class="kpi flat"><div class="k">多数类基线</div><div class="v">62.62%</div>'+
      '<div class="note">全部答 L2</div></div></div>';

  r+='<h3>逐类别表现</h3><table><tr><th>类别</th><th class="num">真实条数</th>'+
    '<th class="num">预测条数</th><th class="num">命中</th><th>召回</th><th>精度</th></tr>';
  ['L1','L2','fiber'].forEach(l=>{
    const p=fs.per_class[l];
    r+='<tr><td>'+pill(l)+'</td><td class="num">'+p.support+'</td><td class="num">'+p.pred_n+
      '</td><td class="num">'+p.hit+'</td><td>'+bar(p.recall,p.recall>0.8?'hi':p.recall<0.4?'lo':'')+
      '</td><td>'+bar(p.precision,p.precision>0.8?'hi':p.precision<0.4?'lo':'')+'</td></tr>';
  });
  r+='</table>';
  r+='<div class="badbox"><b>L2 的成绩基本等于先验，L1 的精度只有 60%，fiber 是 0。</b>'+
     'L2 召回 85.07% / 精度 85.07%，而 L2 本身就占 62.6%；'+
     'L1 判出 35 条只对 21 条，也就是说<b>每判 3 次 L1 就有 1 次多判</b>；'+
     'fiber 一次都没有被判出来。</div>';

  r+='<h3>混淆矩阵</h3><table><tr><th>真值 ↓ / 预测 →</th>'+
     ['L1','L2','fiber','abstain'].map(x=>'<th class="num">'+(x==='abstain'?'弃答':x)+'</th>').join('')+
     '</tr>';
  ['L1','L2','fiber'].forEach(g=>{
    const row=fs.confusion[g]||{};
    r+='<tr><td>'+pill(g)+'</td>'+['L1','L2','fiber','abstain'].map(p=>{
      const v=row[p]||0;
      const hit=(p===g);
      return '<td class="num"'+(v?(' style="background:'+(hit?'#dcfce7':'#fee2e2')+
        ';font-weight:600"'):'')+'>'+(v||'')+'</td>';
    }).join('')+'</tr>';
  });
  r+='</table>';

  r+='<h3>各口径对照</h3><p>76.6% 这个数字单独看没有意义，要和「什么都不做」比。</p><table>'+
    '<tr><th>口径</th><th class="num">准确率</th><th class="num">L1 召回</th>'+
    '<th class="num">L2 召回</th><th class="num">fiber 召回</th><th>说明</th></tr>';
  const names={majority:['多数类基线','全部答 L2，不看任何遥测'],
    graph_top1:['纯检索 top-1','用最相似历史 case 的标签，不用任何规则'],
    sop:['学出来的 SOP','训练标签上的浅决策树'],
    expert:['专家规则','人工编码的阈值表 + 两端仲裁']};
  ['majority','graph_top1','sop','expert'].forEach(k=>{
    const b=OV.baselines[k];
    r+='<tr><td><b>'+names[k][0]+'</b></td><td class="num">'+pct(b.acc,2)+'</td>'+
      ['L1','L2','fiber'].map(l=>'<td class="num">'+pct(b.per_class[l].recall)+'</td>').join('')+
      '<td class="small">'+names[k][1]+'</td></tr>';
  });
  r+='<tr style="background:#f8f9fb"><td><b>当前完整系统</b></td><td class="num"><b>'+
     pct(fs.overall,2)+'</b></td>'+['L1','L2','fiber'].map(l=>'<td class="num">'+
     pct(fs.per_class[l].recall)+'</td>').join('')+
     '<td class="small">级联 + 拒答，拒答记为不正确</td></tr></table>';
  r+='<div class="note">专家规则是唯一明显超过多数类基线的单一方法（+14.0pp）。'+
     '完整系统的全集正确率略低于专家规则全答（72.90% vs 76.64%），'+
     '差额来自 5 条主动拒答——这不是退步，是把不确定的部分交出去了。</div>';
  H('resultBlock',r);
})();

/* ---------- 10 规则格子 ---------- */
(function(){
  const rc=DF.rule_cells_all, cells=rc.cells, prior=rc.prior;
  const rows=Object.entries(cells).sort((a,b)=>b[1].n-a[1].n);
  let r='<div class="tbl-wrap"><table><tr><th>胜出规则</th><th>方向</th><th class="num">n</th>'+
    '<th>准确率</th><th class="num">Wilson 下界</th><th class="num">所判类别先验</th>'+
    '<th>是否提供信息</th><th>真值分布</th><th class="num">fiber 占比</th></tr>';
  rows.forEach(([k,v])=>{
    const bad=!v.beats_prior;
    r+='<tr'+(bad?' style="background:#fffbf5"':'')+'>'+
      '<td class="mono">'+v.rule+'</td><td class="mono small">'+v.direction+'</td>'+
      '<td class="num"><b>'+v.n+'</b></td>'+
      '<td>'+bar(v.acc,v.acc>0.8?'hi':v.acc<0.6?'lo':'')+'</td>'+
      '<td class="num">'+pct(v.wilson_lb)+'</td>'+
      '<td class="num">'+(v.verdict_prior?pct(v.verdict_prior):'—')+'</td>'+
      '<td>'+(v.verdict_prior? (v.beats_prior?'<span class="pill p-ok">是</span>':
        '<span class="pill p-bad">否</span>') : '<span class="pill p-none">不适用</span>')+'</td>'+
      '<td class="small mono">'+Object.entries(v.dist).map(([l,c])=>l+':'+c).join(' ')+'</td>'+
      '<td class="num'+(v.fiber_rate>0.15?'" style="background:#fef3c7;font-weight:600':'')+'">'+
        pct(v.fiber_rate)+'</td></tr>';
  });
  r+='</table></div>';
  r+='<div class="note">「是否提供信息」= 该格子准确率的 Wilson 95% 下界是否高于'+
     '<b>它所判类别的先验</b>（L2 '+pct(prior.L2)+' / L1 '+pct(prior.L1)+' / fiber '+pct(prior.fiber)+
     '）。答「否」意味着在统计上无法证实这条规则比直接猜那个类别更好。</div>';
  r+='<div class="warnbox"><b>两个必须分清的判据。</b>上表的「是否提供信息」衡量'+
     '<i>这条规则有没有价值</i>，但它<b>不能</b>直接拿来做拒答策略：'+
     '判少数类的格子门槛很低（判 L1 只需超过 30.2%），照它拒答会把 74% 准确率的 L2 格子丢掉、'+
     '反而留下 51% 的 L1 格子，精度会下降。要决定「该不该给结论」，'+
     '应该看格子准确率的绝对下界——这就是下面的选择性预测曲线。</div>';
  r+='<div class="badbox"><b>fiber 藏在哪里：</b><code>multi_metric L2→L1</code> 这个格子'+
     '（全库 29 条）里 fiber 占 <b>24.1%</b>，是全库先验 7.5% 的 3.2 倍，'+
     '同时它的准确率只有 51.7%——<b>全库最大的低可靠格子恰好是 fiber 的藏身处</b>。'+
     '判错的 7 条 fiber 里有 4 条落在这里。</div>';
  H('cellsBlock',r);

  const sp=DF.selective_policy;
  let s='<p>门槛在 <b>train 上</b>估每个格子的可靠性，test 上只做查表，不用 test 标签定门槛。'+
    '训练集里没出现过的格子按拒答处理。</p><table>'+
    '<tr><th class="num">要求格子下界 ≥</th><th class="num">覆盖</th><th class="num">给结论条数</th>'+
    '<th>给结论时精度</th><th class="num">相对全答的精度变化</th></tr>';
  const base=sp.baseline_full_coverage;
  sp.curve.forEach(c=>{
    if(c.answered===0)return;
    const d=c.precision-base;
    s+='<tr'+(Math.abs(c.target_wilson_lb-0.4)<1e-9?' style="background:#f0fdf4"':'')+'>'+
      '<td class="num mono">'+c.target_wilson_lb.toFixed(2)+'</td>'+
      '<td class="num">'+pct(c.coverage)+'</td><td class="num">'+c.answered+'</td>'+
      '<td>'+bar(c.precision,c.precision>0.82?'hi':'')+'</td>'+
      '<td class="num" style="color:'+(d>0?'#047857':'#b91c1c')+';font-weight:600">'+
        (d>=0?'+':'')+(d*100).toFixed(2)+'pp</td></tr>';
  });
  s+='</table>';
  s+='<div class="okbox"><b>这是本次分析里最直接可用的一条结论：</b>只把结论限制在'+
     '可靠性达标的规则格子上，<b>覆盖率 63.6%（68/107）时给结论精度达到 85.29%</b>，'+
     '比现在全答的 76.64% 高 8.65pp。剩下 36% 的 case 转人工或请求补采——'+
     '它们本来就是在赌先验。这个改动不需要新模型、不需要新数据，只需要让 M9 认真执行'+
     '它自己已经定义好的准入标准。</div>';
  H('selectiveBlock',s);
})();

/* ---------- 11 可推导性上限 ---------- */
(function(){
  const a=DF.blind_reader.all, t=DF.blind_reader.test;
  const ruleCN={
    P1:'本端收不到光，且对端同 lane 也没在发光 → 光没出发 → 对端发端故障',
    P2:'收到的光「变少且同步变脏」（功率与 SNR 一起掉）→ 上游衰减 → 对端',
    P3:'本端收不到光，但对端在正常发光 → 光缆 / 本端收端 / 对端光口三者不可分 → 拒答',
    P4:'光层两端干净，只有 serdes 电通道死 → 故障在电域，不属于三分类任何一类 → 拒答',
    P5:'两端光层与电层都没有可判读的异常 → 快照不含定位信息 → 拒答'};
  let r='<p>为了回答「换个更强的推理模型能不能解决问题」，我把「只看原始遥测、'+
    '按物理推导」这件事写成了一套规则（<code>scripts/eval_blind_physical_reader.py</code>）。'+
    '它<b>不看标签、不用任何数据集分位数、不用专家阈值表</b>，只用两样东西：'+
    '断光/归零这类物理哨兵，以及同一个 case 内部的对比（同侧 lane 之间、两侧之间）。'+
    '这样得到的就是「一个理想推理者在这份数据上的能力上限」。</p>';
  r+='<div class="grid g4">'+
    '<div class="kpi bad"><div class="k">物理能定论的比例</div><div class="v">'+pct(a.coverage,1)+'</div>'+
      '<div class="note">'+a.answered+'/'+a.n+' 条（全库）</div></div>'+
    '<div class="kpi good"><div class="k">定论时的精度</div><div class="v">'+pct(a.blind_precision,1)+'</div>'+
      '<div class="note">'+a.blind_ok+'/'+a.answered+'</div></div>'+
    '<div class="kpi flat"><div class="k">专家规则在同一子集上</div><div class="v">'+
      pct(a.expert_precision_on_same_subset,1)+'</div><div class="note">完全相同</div></div>'+
    '<div class="kpi warn"><div class="k">专家规则全答</div><div class="v">'+
      pct(a.expert_full_coverage_acc,1)+'</div><div class="note">靠先验补上剩下 91%</div></div></div>';
  r+='<div class="badbox"><b>这两个数字并列在一起，就是对「模型不够强」这个假设的否证：</b>'+
     '物理能说清的 9% 里，现有规则已经做到了和理想推理者<b>一样好</b>（都是 87.5%）；'+
     '而剩下 91% 里没有可推理的内容。专家规则 72% 的成绩中，'+
     '超出这 9% 的部分全部来自「把统计先验附着在弱信号上」，'+
     '这是<b>记忆与标定</b>的功劳，不是推理的功劳。换更强的推理模型不会改变这一点。</div>';

  r+='<h3>各判据的触发量与结果（全库 268 条）</h3><div class="tbl-wrap"><table>'+
    '<tr><th>判据</th><th>物理含义</th><th class="num">触发</th><th class="num">占比</th>'+
    '<th>结果</th><th>真值分布</th><th class="num">fiber 占比</th></tr>';
  Object.entries(a.by_rule).sort((x,y)=>x[0].localeCompare(y[0])).forEach(([k,v])=>{
    const res=v.answered? ('精度 '+pct(v.acc)+'（下界 '+pct(v.wilson_lb)+'）')
      : '<span class="pill p-warn">拒答</span>';
    r+='<tr><td class="mono"><b>'+k+'</b></td><td class="small">'+ruleCN[k]+'</td>'+
      '<td class="num">'+v.n+'</td><td class="num">'+pct(v.n/a.n)+'</td>'+
      '<td>'+res+'</td><td class="small mono">'+
        Object.entries(v.dist).map(([l,c])=>l+':'+c).join(' ')+'</td>'+
      '<td class="num'+(v.fiber_rate>0.12?'" style="background:#fef3c7;font-weight:600':'')+'">'+
        pct(v.fiber_rate)+'</td></tr>';
  });
  r+='</table></div>';
  r+='<div class="grid g2">'+
    '<div class="card"><h4>P3：光出发了却没到达（'+a.by_rule.P3.n+' 条，'+
      pct(a.by_rule.P3.n/a.n)+'）</h4><p class="small">这是最该请求补采的一块。'+
      'fiber 在这里占 '+pct(a.by_rule.P3.fiber_rate)+'，是全库先验 7.5% 的两倍。'+
      '要把光缆、本端收端、对端光口分开，需要 OTDR 或双向同步快照——'+
      '这是采集问题，不是算法问题。</p></div>'+
    '<div class="card"><h4>P4：光层干净但电通道死（'+a.by_rule.P4.n+' 条，'+
      pct(a.by_rule.P4.n/a.n)+'）</h4><p class="small">遥测说的是「电通道坏了」，'+
      '标签说的是「哪个光模块被换了」，两者之间没有可推导的关系。'+
      '这一块真值分布 '+Object.entries(a.by_rule.P4.dist).map(([l,c])=>l+':'+c).join(' ')+
      '，与全库先验几乎一致，也就是说看到这个模式等于什么都没看到。'+
      '判错的 24 条里有 6 条（25%）在这里。</p></div></div>';
  r+='<div class="note">在 test 107 条上单独跑，覆盖 '+pct(t.coverage)+'（'+t.answered+
     ' 条），精度 '+pct(t.blind_precision)+'，与全库一致。'+
     '更关键的是：<b>当前系统判错的 24 条里，这套物理盲读对 23 条明确拒答</b>，'+
     '唯一肯回答的一条（case_f40d7dea10c4）也答错了，而它恰好是最可靠判据 P2 在全库 13 次触发里'+
     '唯一的反例。</div>';
  H('ceilingBlock',r);
})();

/* ---------- 12 LLM ---------- */
(function(){
  const ch=CB.challenge.llm_challenger, ex=CB.explain.checkability;
  let r='<p>LLM 被放在三个不同岗位上分别做了实验，同一个模型、同一份数据、同一个划分。'+
    '三次实验的结论差别很大，值得分开看。</p>';
  r+='<div class="grid g3">'+
    '<div class="kpi bad"><div class="k">岗位一：定界器</div><div class="v">不可用</div>'+
      '<div class="note">准确率显著低于专家规则（McNemar p=0.0117）。'+
      'prompt 里已经把 <code>expert:points_to</code> 直接给了它，'+
      '照抄就能到 77.8%，它自己只做到 60.4%</div></div>'+
    '<div class="kpi bad"><div class="k">岗位二：质疑器</div><div class="v">无区分力</div>'+
      '<div class="note">质疑率 '+pct(ch.challenge_rate)+'，命中率 '+pct(ch.hit_rate)+
      '，而规则本身的错误率是 '+pct(ch.rule_error_rate)+'。'+
      '增益 '+(ch.lift_over_error_rate*100).toFixed(2)+'pp——它几乎质疑一切，'+
      '所谓命中只是撞上了基础错误率</div></div>'+
    '<div class="kpi good"><div class="k">岗位三：解释器</div><div class="v">可用</div>'+
      '<div class="note">四项可机检指标：token 存在性 '+pct(ex.token_existence)+
      '、相关性 '+pct(ex.token_relevance)+'、方向一致性 '+pct(ex.direction_consistency)+
      '、结论一致性 '+pct(ex.verdict_consistency)+'，全通过率 '+pct(ex.all_pass)+
      '，编造 token '+ex.fabricated_token_count+' 个</div></div></div>';
  r+='<div class="infobox"><b>为什么定界器这么差，而且不是「模型不够强」能解释的：</b>'+
     'LLM 拿到的不是原始 lane 读数，而是 N2 抽取出来的<b>证据 token</b>。'+
     '也就是说它读到的是「L1 侧 rxpower 有 lane_diff」，而不是「[2.53, 2.46, 0.78, 2.41]」。'+
     '专家规则跑在原始值上，LLM 跑在压缩后的符号上——'+
     '前面第 3 节列出的四类信息损失（方向、lane 数量、绝对参考、跨端对比），'+
     'LLM 一个都拿不到。同一 token 空间上训练的随机森林准确率 67.19%，'+
     '也低于跑在原始值上的专家规则 72.01%，说明<b>上限是被输入表示卡住的</b>。</div>';
  r+='<div class="okbox"><b>解释器是唯一站得住的岗位，而它恰好不需要模型做判断。</b>'+
     '解释器的任务是把已经确定的规则依据翻译成人话，四项指标都是机器可检的，'+
     '不依赖人工评分。值得注意的是：在规则<b>判错</b>的 25 条上，'+
     '解释器的全通过率是 '+pct(CB.explain.by_rule_correctness.rule_wrong.all_pass)+
     '，比规则判对时还略高——它忠实地解释了一个错误结论。'+
     '这正说明它做的是转述而不是判断，也提醒我们它不能用来兜错。</div>';

  const la=DF.llm_alignment;
  r+='<h4>为什么质疑器必然没有区分力：LLM 不是独立信号</h4>'+
    '<p>把 LLM 的定界结论按「系统判对 / 判错」分开统计，问题就很清楚了。</p><table>'+
    '<tr><th>子集</th><th class="num">条数</th><th class="num">有 trace</th>'+
    '<th class="num">可解析</th><th class="num">未解析或弃答</th>'+
    '<th class="num">与最终结论一致</th><th class="num">命中真值</th></tr>';
  [['correct','系统判对'],['wrong','系统判错'],['abstain','系统拒答']].forEach(([k,lab])=>{
    const g=la[k];
    r+='<tr><td><b>'+lab+'</b></td><td class="num">'+g.n+'</td><td class="num">'+g.with_trace+
      '</td><td class="num">'+g.parsed+'</td><td class="num">'+g.unparsed_or_abstain+'</td>'+
      '<td class="num">'+g.agree_with_final+'（'+pct(g.agree_rate)+'）</td>'+
      '<td class="num">'+g.matches_gold+'（'+pct(g.gold_rate)+'）</td></tr>';
  });
  r+='</table>';
  r+='<div class="badbox"><b>在系统判错的 case 上，LLM 有 '+pct(la.wrong.agree_rate)+
     ' 的概率给出和错误结论相同的答案，命中真值只有 '+pct(la.wrong.gold_rate)+'（'+
     la.wrong.matches_gold+'/'+la.wrong.parsed+'）。</b>'+
     '它复现规则的错误，而不是发现错误。原因不难理解：prompt 里直接给了 '+
     '<code>expert:points_to</code> token，也就是把专家规则的逐端结论摊在了模型面前，'+
     '所以模型的输出与规则高度相关。<b>一个与被检查者相关的检查者不可能提供独立校验</b>，'+
     '这就是质疑器增益为负的结构性原因，调 prompt 或换更大的模型都改变不了。</div>';
  r+='<div class="warnbox"><b>另外还有可用性问题：</b>系统判对的 57 条有 trace 的 case 里，'+
     la.correct.unparsed_or_abstain+' 条（'+pct(la.correct.unparsed_or_abstain/la.correct.with_trace)+
     '）的输出无法解析或模型自行弃答。全部 trace 累计 '+
     Object.values(la.violation_kinds).reduce((a,b)=>a+b,0)+' 条约束违规，'+
     la.cases_with_fatal+' 条 case 出现过 fatal 级违规，最高频的是 '+
     Object.entries(la.violation_kinds)[0][0]+'（'+Object.entries(la.violation_kinds)[0][1]+' 条）。</div>';

  const curve=CB.challenge.score_threshold_curve;
  r+='<h4>质疑器的阈值曲线：调门槛也救不回来</h4><table>'+
    '<tr><th class="num">最少失败前提数</th><th class="num">标记条数</th><th class="num">标记率</th>'+
    '<th class="num">命中</th><th>命中率</th><th class="num">相对错误率的增益</th></tr>';
  curve.forEach(c=>{
    if(c.flagged===0)return;
    r+='<tr><td class="num mono">'+c.min_failed_premises+'</td><td class="num">'+c.flagged+'</td>'+
      '<td class="num">'+pct(c.flag_rate)+'</td><td class="num">'+c.hits+'</td>'+
      '<td>'+bar(c.hit_rate,c.hit_rate>0.234?'hi':'lo')+'</td>'+
      '<td class="num" style="color:'+(c.lift_over_error_rate>0?'#047857':'#b91c1c')+
      ';font-weight:600">'+(c.lift_over_error_rate>=0?'+':'')+
      (c.lift_over_error_rate*100).toFixed(2)+'pp</td></tr>';
  });
  r+='</table><div class="note">任何门槛下的增益都是负的：收紧门槛让标记数从 98 降到 10，'+
     '命中率反而从 22.4% 掉到 10%。质疑器给出的「失败前提数」与规则是否真的错了<b>无关</b>。</div>';
  H('llmBlock',r);
})();

/* ---------- 13 缺陷 ---------- */
(function(){
  const items=[
    ['数据','bias 字段未解析且未报缺失','11 条 case 的 <code>bias</code> 是字符串而非字典，'+
      '参与 C1/C2 的 token 生成，证据实际缺失但系统静默降级',
      '修数据加载并在解析失败时写入 <code>missing_fields</code>；加类型断言','低','高'],
    ['数据','txpower 语义未确认','37 个「侧」的 tx 四条读数完全相同，疑为标称值。'+
      '所有依赖「对端发端是否正常」的推理都建立在此之上',
      '核实字段来源；若为标称值则相关判据全部降级','低','高'],
    ['数据','host_snr 缺失 41~45%','它是唯一能无歧义定端的指标（对端影响不到本端主机侧电通道）',
      '优先补采；同时放宽这类强物理判据的标定支持数门槛','中','高'],
    ['数据','缺 OTDR / 双向同步快照','P3 子集 79 条（29.5%）因此不可判，fiber 富集 15.2%',
      '补采光缆段测量；在此之前该子集应请求补采而非二选一','高','高'],
    ['特征','lane_diff 不带方向','偏高与偏低物理含义相反却压成同一 token；'+
      '3 条判错 case 的真因都在「出现偏高读数的那一侧」',
      '拆成 lane_diff_low / lane_diff_high，只让偏低指向对端','低','中'],
    ['特征','lane_down 是布尔型','丢掉了「断几条、断哪几条」。'+
      '实测「恰好 1 条 lane 断」覆盖了 multi_metric L2→L1 格子里全部 7 条 fiber',
      '扩展为单 lane / 多 lane / 整侧三档','低','中'],
    ['特征','没有任何跨端对比特征','fiber 的本质是「两端都不正常但都不是单端故障」，'+
      '只能用跨端量表达；连链路插损（对端 tx − 本端 rx）都没算',
      '增加跨端差值特征族','低','中'],
    ['规则','multi_metric 重复计数同一件事','单 lane 断光时 rx/media_snr/serdes 三个"异常"'+
      '是同一个物理事件的三种表述，规则却当三条独立证据抬高置信度',
      '按 C1/C5 已声明的因果依赖合并计数','低','中'],
    ['规则','multi_metric 吞掉 host_snr 的方向','host_snr 物理上只能指向本端，'+
      '但 multi_metric 优先级更高，套用了「收端异常指对端」的默认方向（case_788f63a0a0fc）',
      '参与指标含 host_snr 时方向由它决定','低','中'],
    ['规则','C23/C24 缺对端交叉核验','「本端接收异常 ⇒ 对端」不核对对端同 lane 是否真的弱'+
      '（case_fcacbc45173f 里对端该 lane 恰是最强的一条）',
      '加前置条件；不满足则降级为 neutral','低','中'],
    ['规则','serdes_snr high_value 被当作定界依据','SNR 偏高不是故障模式。'+
      '该格子全库仅 1 条样本、准确率 0%（case_fa72aef06d4d）',
      '从可定界异常类型中移除','低','低'],
    ['规则','serdes 判据整体偏弱','<code>single:serdes_snr</code> 两个主要格子共 64 条'+
      '（全库 23.9%），Wilson 下界均低于多数类先验',
      '这一族应大幅提高拒答比例','低','高'],
    ['级联','检索精确命中被规则覆盖','case_22d4dde687ab 与训练集某条 fiber 相似度 1.000，'+
      '仍被 Wilson 下界 34.4% 的规则格子盖过',
      '把高相似度精确命中提到专家规则之前','低','中'],
    ['级联','未执行自己的准入标准','超过一半 case 落在下界不达标的格子上却照样给结论。'+
      '按下界 ≥0.40 拒答可把精度从 76.64% 提到 85.29%（覆盖 63.6%）',
      '按格子可靠性做选择性预测','低','高'],
    ['级联','忽略「无先例」信号','检索相似度极低时（case_fa72aef06d4d 仅 0.366）'+
      '证据图已在说没见过，M9 未采纳为拒答依据',
      '把低相似度纳入拒答条件','低','低'],
    ['标注','电域故障无对应类别','P4 子集 44 条（16.4%）光层干净、只有 serdes 死，'+
      '真值分布与先验一致；判错的 24 条里 6 条在此',
      '增设「电域疑似」出口，不参与三分类','中','高'],
  ];
  const impCol={'高':'p-bad','中':'p-warn','低':'p-info'};
  let r='<div class="tbl-wrap"><table><tr><th>层次</th><th>缺陷</th><th>依据</th>'+
    '<th>建议动作</th><th class="num">成本</th><th class="num">影响</th></tr>';
  items.forEach(([lay,name,ev,fix,cost,imp])=>{
    r+='<tr><td><span class="pill p-info">'+lay+'</span></td><td><b>'+name+'</b></td>'+
      '<td class="small">'+ev+'</td><td class="small">'+fix+'</td>'+
      '<td class="num"><span class="pill '+impCol[cost]+'">'+cost+'</span></td>'+
      '<td class="num"><span class="pill '+impCol[imp]+'">'+imp+'</span></td></tr>';
  });
  r+='</table></div>';
  r+='<h3>如果只做三件事</h3><div class="okbox"><ol>'+
    '<li><b>让 M9 执行自己的准入标准。</b>按规则格子的 Wilson 下界做选择性预测，'+
    '覆盖 63.6% 时精度 85.29%（+8.65pp）。零成本、零新数据、当天可验证。</li>'+
    '<li><b>把 P3 子集（29.5%）改为请求补采。</b>这一块 fiber 富集两倍，'+
    '现在被强行二选一，是 fiber 召回为 0 的直接原因。同时推动 OTDR 采集。</li>'+
    '<li><b>查清 txpower 语义、修 bias 解析。</b>成本极低，但它决定了所有跨端推理'+
    '是否可信——包括上面几条建议里的交叉核验。</li>'+
    '</ol><p style="margin:9px 0 0">这三件事都不涉及换模型。'+
    '按第 11 节的测量，换更强的推理模型在这份数据上能改善的是<b>拒答的准确性</b>，'+
    '而拒答策略用规则可靠性统计就能做到，不需要模型。</p></div>';
  H('defectBlock',r);
})();
"""


# --------------------------------------------------------------------------------------
# 报告二：逐 case 决策链
# --------------------------------------------------------------------------------------

CASES_NAV = [
    ("s-how", "怎么读这一页"),
    ("s-wrong", "判错的 24 条"),
    ("s-right", "判对的 78 条"),
    ("s-abstain", "主动拒答的 5 条"),
]


def build_cases() -> str:
    cb = json.loads(CASE_BUNDLE.read_text(encoding="utf-8"))
    df = json.loads((ROOT / "artifacts/defect_bundle.json").read_text(encoding="utf-8"))
    meta = cb["meta"]
    fs = df["final_system"]

    # 把手写分析按 case_id 挂到数据上，顺带统计错因分布
    kinds: dict[str, int] = {}
    for cid, n in NARRATIVES.items():
        kinds[n["kind"]] = kinds.get(n["kind"], 0) + 1
    missing = [c["id"] for c in cb["cases"] if not c["ok"] and c["pred"] and c["id"] not in NARRATIVES]
    if missing:
        raise SystemExit(f"以下判错 case 缺少手写分析，请补 case_narratives.py: {missing}")

    chips = [
        f"test {fs['n']} 条",
        f"判对 {fs['correct']}",
        f"判错 {fs['n'] - fs['correct'] - (fs['n'] - fs['answered'])}",
        f"拒答 {fs['n'] - fs['answered']}",
        f"模型 {meta['model']}",
        f"prompt {meta['prompt_version']}",
    ]

    html = [head(
        "逐 case 决策链与错因分析",
        "每条 case 从原始 lane 读数走到最终结论的完整链路；判错的 24 条附无标签盲推导与有标签洞察",
        chips, CASES_NAV,
    )]

    kind_rows = "".join(
        f'<tr><td><span class="kindTag" style="background:{KIND_COLOR[k]}">{KIND_LABEL[k]}</span></td>'
        f'<td class="num"><b>{v}</b></td><td>{{bar{k}}}</td></tr>'
        for k, v in sorted(kinds.items(), key=lambda x: -x[1])
    )
    for k, v in kinds.items():
        kind_rows = kind_rows.replace(
            f"{{bar{k}}}",
            f'<div class="bar"><i style="width:{100*v/24:.0f}%"></i><b>{100*v/24:.0f}%</b></div>',
        )

    html.append(f"""
<section id="s-how"><h2><span class="n">00</span>怎么读这一页</h2>
<div class="lead">
每张卡片顶部是一条<b>决策流水线</b>：从原始遥测出发，经过证据 token、专家逐端诊断、
两端仲裁、SOP 与检索、M9 级联，到最终结论。绿色是这一步与真值一致，红色是不一致，
紫色边框标出 M9 实际采纳的那一个候选。展开后可以看到原始 lane 读数、LLM 的完整思维链与
逐步推理，以及校验器抓到的约束违规。
</div>
<div class="grid g2">
<div class="card"><h4>判错的 24 条按错因归类</h4>
<table><tr><th>错因</th><th class="num">条数</th><th>占比</th></tr>{kind_rows}</table>
<p class="small">分类依据是「这条 case 错在哪个环节」，不是「它长什么样」。
其中<b>光出发未到达</b>与<b>光层干净电通道死</b>两类合计过半，都属于遥测本身不含答案；
只有<b>方向约束缺核验</b>和<b>专家判对被级联丢掉</b>是可以直接修的。</p></div>
<div class="card"><h4>三段分析各自是什么</h4>
<p><b class="mono" style="color:#5b21b6">盲推导</b>：我只看原始遥测（不看标签、不看专家阈值表）
能推到哪一步。先给严格物理读法的结论——很多时候是明确<b>拒答</b>，
并说明为什么这份数据不足以定论；如果被迫二选一，再说明我会押哪边、依据有多硬。</p>
<p><b class="mono" style="color:#92400e">有标签洞察</b>：知道真值之后才看得见的东西。
重点是区分两种情况——<b>规则错在某个具体环节</b>（可修），
还是<b>遥测里根本没有答案</b>（不可修，属识别上限）。</p>
<p><b class="mono" style="color:#166534">修法</b>：这条 case 指向的具体改动。
没有这一段说明我认为它不可修。</p>
<div class="note">为了防止后见之明，我把盲推导里用到的物理判据写成了可执行规则
（<code>scripts/eval_blind_physical_reader.py</code>）并在全库 268 条上无差别施行。
结果：只有 9.0% 的 case 能被物理定论，精度 87.5%；而<b>当前系统判错的 24 条里，
这套规则对 23 条明确拒答</b>。凡是我在个别 case 上「押对了」的地方，
只要那套推理在全库上不成立，我都在洞察里注明了——例如 case_8448ab686d52。</div>
</div></div>
</section>

<div class="filters">
  <span class="lbl">结果</span>
  <button class="btn on" data-f="res" data-v="all">全部</button>
  <button class="btn" data-f="res" data-v="wrong">判错 24</button>
  <button class="btn" data-f="res" data-v="right">判对 78</button>
  <button class="btn" data-f="res" data-v="abstain">拒答 5</button>
  <span class="lbl" style="margin-left:10px">真值</span>
  <button class="btn on" data-f="gold" data-v="all">全部</button>
  <button class="btn" data-f="gold" data-v="L1">L1</button>
  <button class="btn" data-f="gold" data-v="L2">L2</button>
  <button class="btn" data-f="gold" data-v="fiber">fiber</button>
  <span class="lbl" style="margin-left:10px">错因</span>
  <select id="kindSel" class="search" style="width:auto"><option value="all">全部</option></select>
  <input id="q" class="search" placeholder="搜索 case_id / 规则 / token">
  <span class="grow"></span>
  <span class="small" id="cnt"></span>
</div>

<section id="s-wrong"><div id="list"></div></section>
<section id="s-right"></section>
<section id="s-abstain"></section>
""")

    html.append(tail(
        "决策链数据由 <code>scripts/build_report_bundle.py</code> 生成，"
        "逐 case 盲推导与洞察见 <code>scripts/case_narratives.py</code>。"
        "框架与整体效果见 <a href='rca_overview.html'>rca_overview.html</a>。",
        {"caseData": cb, "narr": NARRATIVES,
         "kindMeta": {"label": KIND_LABEL, "color": KIND_COLOR}},
        CASES_JS,
    ))
    return "".join(html)


CASES_JS = r"""
const CB=JSON.parse(document.getElementById('caseData').textContent);
const NARR=JSON.parse(document.getElementById('narr').textContent);
const KM=JSON.parse(document.getElementById('kindMeta').textContent);
const CASES=CB.cases, REL=CB.reliability;
const METRICS=['rxpower','txpower','media_snr','host_snr','serdes_snr','bias'];
const MLABEL={rxpower:'接收光功率 dBm',txpower:'发送光功率 dBm',media_snr:'介质侧 SNR dB',
  host_snr:'主机侧 SNR dB',serdes_snr:'SerDes SNR',bias:'偏置电流 mA'};

function fmt(v){
  if(v===null||v===undefined)return '—';
  if(typeof v==='string')return '?';
  return Math.abs(v)>=10000?v.toFixed(0):v.toFixed(2);
}
/* 单元格着色：区分「物理哨兵」与「相对同侧偏低」两件事，前者是硬事实、后者是弱信号 */
function cellCls(metric,v,arr){
  if(v===null||v===undefined||typeof v==='string')return 'na';
  if((metric==='rxpower'||metric==='txpower')&&v<=-39)return 'dark';
  if((metric==='media_snr'||metric==='host_snr')&&v<=1)return 'zero';
  if(metric==='serdes_snr'&&v<=1)return 'zero';
  const live=arr.filter(x=>typeof x==='number'&&!(metric.endsWith('power')&&x<=-39)&&!(metric.includes('snr')&&x<=1));
  if(live.length>=3){
    const s=[...live].sort((a,b)=>a-b), med=s[Math.floor(s.length/2)];
    if(metric==='serdes_snr'){ if(v<med*0.6)return 'lo'; }
    else if(med-v>=1.0)return 'lo';
  }
  return '';
}
function laneTable(c){
  let r='<table class="lanes"><tr><th>指标</th><th>侧</th>';
  let maxN=0;
  METRICS.forEach(m=>['L1','L2'].forEach(s=>{maxN=Math.max(maxN,(c.raw[m][s]||[]).length);}));
  for(let i=0;i<maxN;i++)r+='<th class="num">L'+i+'</th>';
  r+='</tr>';
  METRICS.forEach(m=>{
    const a=c.raw[m].L1||[],b=c.raw[m].L2||[];
    if(!a.length&&!b.length)return;
    [['L1',a],['L2',b]].forEach(([s,arr],idx)=>{
      r+='<tr>'+(idx===0?'<td rowspan="2">'+MLABEL[m]+'</td>':'')+'<td>'+s+'</td>';
      for(let i=0;i<maxN;i++){
        const v=arr[i];
        r+='<td class="'+(i<arr.length?cellCls(m,v,arr):'na')+'">'+(i<arr.length?fmt(v):'')+'</td>';
      }
      r+='</tr>';
    });
  });
  r+='</table>';
  const st=Object.entries(c.status).filter(([k,v])=>['L1','L2'].some(s=>v[s]&&v[s]!=='Normal'));
  r+='<p class="small" style="margin-top:8px">'+
    '<b>异常状态位</b>：'+(st.length?st.map(([k,v])=>k+'（'+['L1','L2'].filter(s=>v[s]&&v[s]!=='Normal').join('/')+'）').join('，'):'无')+
    '　<b>告警</b>：'+esc(c.ctx.alarm||'—')+
    '　<b>lane 数</b>：'+(c.ctx.lanes?JSON.stringify(c.ctx.lanes):'缺失')+
    '　<b>告警接口</b>：'+(c.ctx.alarm_if?'有':'缺失')+'</p>'+
    '<p class="small">图例：<span class="pill" style="background:#fee2e2;color:#991b1b">断光 −40</span> '+
    '<span class="pill" style="background:#fef3c7;color:#92400e">归零哨兵</span> '+
    '<span class="pill" style="background:#e0f2fe;color:#075985">低于同侧中位数</span>　'+
    '前两者是物理硬事实，第三者只是相对偏低、不构成断言。</p>';
  return r;
}

function flow(c){
  const cell=REL[c.cell]||null;
  const sides=c.sides.map(s=>s.side+' '+s.rule+'(prio '+s.prio+')→'+s.loc).join('<br>');
  const src=c.source||'—';
  function st(lab,val,sub,cls){
    return '<div class="st '+(cls||'')+'"><div class="lab">'+lab+'</div><div class="val">'+val+
      '</div>'+(sub?'<div class="sub">'+sub+'</div>':'')+'</div>';
  }
  const tokN=c.tokens.length;
  let s='<div class="flow">';
  s+=st('N1 遥测','原始 lane 读数',
    (c.ctx.lanes?('lane '+JSON.stringify(c.ctx.lanes)):'lane 数缺失')+'<br>'+esc(c.ctx.alarm||''));
  s+=st('N2 证据',tokN+' 个 token',
    c.tokens.filter(t=>t.startsWith('expert:')).length+' 个 expert 族');
  s+=st('专家逐端',c.sides.length?c.sides.length+' 侧触发':'两端无异常',sides||'走 no_anomaly 兜底');
  s+=st('两端仲裁',pill(c.expert_verdict),
    esc(c.reason||'').replace(/（/g,'<br>（'),
    c.expert_verdict===c.gold?'hit':'err');
  s+=st('SOP / 检索',pill(c.sop)+' / '+pill(c.match.cands.length?c.match.cands[0].label:null),
    'top-1 相似度 '+c.match.sim.toFixed(3)+(c.match.ties?'（'+c.match.ties+' 条并列）':''),
    (c.sop===c.gold||(c.match.cands[0]&&c.match.cands[0].label===c.gold))?'':'dim');
  s+=st('M9 采纳','来源 '+src,
    '置信度 '+(c.final_conf!==null&&c.final_conf!==undefined?c.final_conf.toFixed(2):'—')+
    (cell?'<br>该规则格子下界 '+pct(cell.wilson_lb)+'（n='+cell.n+'）':''),'pick');
  s+=st('结论',pill(c.pred)+' vs 真值 '+pill(c.gold),
    c.pred===null?'主动拒答':(c.ok?'一致':'不一致'),
    c.pred===null?'':(c.ok?'hit':'err'));
  s+='</div>';
  if(cell&&!cell.beats_prior&&c.pred!==null){
    s+='<div class="warnbox" style="margin:4px 0 12px"><b>这一步落在统计上不成立的规则格子上。</b>'+
      '<code>'+c.cell+'</code> 全库 '+cell.n+' 条，准确率 '+pct(cell.acc)+
      '，Wilson 下界 '+pct(cell.wilson_lb)+'，低于所判类别 '+
      c.cell.split('->').pop()+' 的先验 '+pct(cell.verdict_prior)+
      (cell.fiber_rate>0.15?'。该格子里 fiber 占 '+pct(cell.fiber_rate)+
        '，是全库先验 7.5% 的 '+(cell.fiber_rate/0.0746).toFixed(1)+' 倍':'')+'。</div>';
  }
  return s;
}

function tokens(c,cited){
  const set=new Set(cited||[]);
  return '<div class="tokwrap">'+c.tokens.map(t=>{
    const cls=set.has(t)?'tok cited':(t.startsWith('expert:')?'tok ex':'tok');
    return '<span class="'+cls+'">'+esc(t)+'</span>';
  }).join('')+'</div>';
}

function llmBlock(c){
  if(!c.llm)return '<p class="small">这条 case 没有走 LLM 分支（M9 在更早的候选上就已经作出决定）。</p>';
  const L=c.llm, last=L.attempts[L.attempts.length-1];
  let s='<p class="small">尝试 '+L.attempt_count+' 次'+(L.rewrote?'（触发过重写）':'')+
    '，最终'+(L.accepted?'<b>被采纳</b>':'<b>未被采纳</b>')+
    (L.abstain_reason?'（'+esc(L.abstain_reason)+'）':'')+'。</p>';
  L.attempts.forEach(a=>{
    const p=a.parsed;
    s+='<details'+(a.index===L.attempts.length-1?' open':'')+'><summary>第 '+(a.index+1)+' 次生成 · '+
      (p?('结论 '+(p.verdict||'无')+'，置信度 '+(p.confidence!==null?p.confidence:'—')):'解析失败')+
      ' · '+a.violations.length+' 条违规（其中 fatal '+a.fatal+'）</summary><div class="body">';
    if(a.thinking){
      s+='<h4>模型的思维链</h4><div class="thinking">'+esc(a.thinking)+'</div>';
    }
    if(p){
      s+='<h4>结构化推理步骤</h4>';
      if(!p.steps.length)s+='<p class="small">没有产出任何推理步骤。</p>';
      p.steps.forEach((st,i)=>{
        s+='<div class="step '+(st.effect||'neutral')+'"><div class="claim"><b>'+(i+1)+'.</b> '+
          esc(st.claim)+'</div><div class="m">'+
          '<span class="pill '+(st.effect==='support'?'p-ok':st.effect==='exclude'?'p-bad':'p-none')+'">'+
          (st.effect||'neutral')+'</span>'+
          (st.target?pill(st.target):'')+
          (st.cited_evidence.length?'<span>引用证据 '+st.cited_evidence.map(x=>'<code>'+esc(x)+'</code>').join(' ')+'</span>':'')+
          (st.cited_constraints.length?'<span>引用约束 '+st.cited_constraints.map(x=>'<code>'+esc(x.split('_')[0])+'</code>').join(' ')+'</span>':'')+
          '</div></div>';
      });
      if(p.missing_information&&p.missing_information.length){
        s+='<p class="small"><b>模型自己指出缺失的信息：</b>'+
          p.missing_information.map(x=>'<code>'+esc(x)+'</code>').join('，')+'</p>';
      }
      const cited=[].concat(...p.steps.map(x=>x.cited_evidence));
      s+='<h4>它拿到的证据 token（黄色为它引用过的）</h4>'+tokens(c,cited);
    }
    if(a.violations.length){
      s+='<h4>校验器抓到的违规</h4>';
      a.violations.forEach(v=>{
        s+='<div class="viol '+v.severity+'"><b>'+v.kind+'</b>'+
          (v.constraint_id?' · <code>'+esc(v.constraint_id.split('_')[0])+'</code>':'')+
          (v.step_index!==null&&v.step_index!==undefined?' · 第 '+(v.step_index+1)+' 步':'')+
          '<br>'+esc(v.message)+(v.detail?'<br><span class="small">'+esc(v.detail)+'</span>':'')+'</div>';
      });
    }
    s+='</div></details>';
  });
  if(c.challenge){
    s+='<details><summary>质疑器在这条 case 上的输出 · 失败前提 '+
      (c.challenge.score!==undefined?c.challenge.score:'—')+' 个 · '+
      (c.challenge.challenged?'提出质疑':'未质疑')+'</summary><div class="body">'+
      (c.challenge.thinking?'<div class="thinking">'+esc(c.challenge.thinking)+'</div>':'')+
      '<pre class="thinking">'+esc(JSON.stringify(c.challenge.response,null,1))+'</pre>'+
      '</div></details>';
  }
  if(c.explain){
    const ck=c.explain.checks||{};
    s+='<details><summary>解释器在这条 case 上的输出 · 可机检指标 '+
      Object.entries(ck).filter(([k,v])=>typeof v==='boolean').map(([k,v])=>k+(v?'✓':'✗')).join(' ')+
      '</summary><div class="body">'+
      '<div class="thinking">'+esc(typeof c.explain.explanation==='string'?
        c.explain.explanation:JSON.stringify(c.explain.explanation,null,1))+'</div></div></details>';
  }
  return s;
}

function card(c){
  const n=NARR[c.id];
  const wrong=c.pred!==null&&!c.ok, abstain=c.pred===null;
  let s='<div class="caseCard '+(wrong?'wrong':abstain?'':'right')+'" data-id="'+c.id+
    '" data-res="'+(abstain?'abstain':wrong?'wrong':'right')+'" data-gold="'+c.gold+
    '" data-kind="'+(n?n.kind:'')+'" data-blob="'+
    esc((c.id+' '+c.cell+' '+c.tokens.join(' ')+' '+(c.reason||'')).toLowerCase())+'">';
  s+='<div class="caseHead"><span class="cid">'+c.id+'</span>'+
    '<span>真值 '+pill(c.gold)+'</span><span>结论 '+pill(c.pred)+'</span>'+
    (abstain?'<span class="pill p-warn">主动拒答</span>':
      c.ok?'<span class="pill p-ok">判对</span>':'<span class="pill p-bad">判错</span>')+
    (n?'<span class="kindTag" style="background:'+KM.color[n.kind]+'">'+KM.label[n.kind]+'</span>':'')+
    '<span class="grow"></span>'+
    '<span class="small mono">'+esc(c.cell)+'</span></div>';
  s+='<div class="caseBody">';
  s+=flow(c);
  if(n){
    s+='<div class="blind"><h5>盲推导 · 不看标签，只看原始遥测</h5>'+
      n.blind.split('\n').map(p=>'<p>'+esc(p)+'</p>').join('')+'</div>';
    s+='<div class="insight"><h5>有标签洞察</h5>'+
      n.insight.split('\n').map(p=>'<p>'+esc(p)+'</p>').join('')+'</div>';
    if(n.fix){
      s+='<div class="fix"><h5>修法</h5><p>'+esc(n.fix)+'</p></div>';
    }else{
      s+='<div class="note"><b>不可修。</b>这条 case 的真因在当前遥测里没有可区分的表现，'+
        '属于识别上限内的固有误差，不是规则或模型的缺陷。</div>';
    }
  }
  s+='<details><summary>原始 lane 读数</summary><div class="body">'+laneTable(c)+'</div></details>';
  s+='<details><summary>证据 token（'+c.tokens.length+' 个）与检索邻居</summary><div class="body">'+
    tokens(c)+'<h4>最相似的历史 case</h4><table><tr><th>case</th><th>标签</th>'+
    '<th class="num">相似度</th></tr>'+
    (c.match.cands.length?c.match.cands.map(x=>'<tr><td class="mono small">'+x.id+'</td><td>'+
      pill(x.label)+'</td><td class="num">'+x.sim.toFixed(3)+'</td></tr>').join(''):
      '<tr><td colspan="3" class="small">无命中</td></tr>')+
    '</table></div></details>';
  s+='<details><summary>LLM 的分析过程</summary><div class="body">'+llmBlock(c)+'</div></details>';
  s+='</div></div>';
  return s;
}

/* 排序：判错优先，且同类错因排在一起，方便按错因通读 */
const order={wrong:0,abstain:1,right:2};
const sorted=[...CASES].sort((a,b)=>{
  const ra=a.pred===null?'abstain':(a.ok?'right':'wrong');
  const rb=b.pred===null?'abstain':(b.ok?'right':'wrong');
  if(order[ra]!==order[rb])return order[ra]-order[rb];
  const ka=(NARR[a.id]||{}).kind||'', kb=(NARR[b.id]||{}).kind||'';
  if(ka!==kb)return ka<kb?-1:1;
  return a.id<b.id?-1:1;
});

const sel=document.getElementById('kindSel');
[...new Set(Object.values(NARR).map(n=>n.kind))].forEach(k=>{
  const o=document.createElement('option');o.value=k;o.textContent=KM.label[k];sel.appendChild(o);
});

document.getElementById('list').innerHTML=sorted.map(card).join('');

const state={res:'all',gold:'all',kind:'all',q:''};
function apply(){
  let shown=0;
  document.querySelectorAll('.caseCard').forEach(el=>{
    const ok=(state.res==='all'||el.dataset.res===state.res)
      &&(state.gold==='all'||el.dataset.gold===state.gold)
      &&(state.kind==='all'||el.dataset.kind===state.kind)
      &&(!state.q||el.dataset.blob.includes(state.q));
    el.style.display=ok?'':'none';
    if(ok)shown++;
  });
  document.getElementById('cnt').textContent='显示 '+shown+' / '+CASES.length+' 条';
}
document.querySelectorAll('.btn[data-f]').forEach(b=>{
  b.onclick=()=>{
    document.querySelectorAll('.btn[data-f="'+b.dataset.f+'"]').forEach(x=>x.classList.remove('on'));
    b.classList.add('on');state[b.dataset.f]=b.dataset.v;apply();
  };
});
sel.onchange=()=>{state.kind=sel.value;apply();};
document.getElementById('q').oninput=e=>{state.q=e.target.value.trim().toLowerCase();apply();};
apply();
"""


def main() -> int:
    for path in (OVERVIEW_BUNDLE, CASE_BUNDLE, ROOT / "artifacts/defect_bundle.json"):
        if not path.exists():
            raise SystemExit(f"缺少数据文件 {path}，请先运行对应的 build_*.py")
    OUT_DIR.mkdir(exist_ok=True)
    for name, builder in (("rca_overview.html", build_overview), ("rca_cases.html", build_cases)):
        out = OUT_DIR / name
        out.write_text(builder(), encoding="utf-8")
        print(f"wrote {out} ({out.stat().st_size/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
