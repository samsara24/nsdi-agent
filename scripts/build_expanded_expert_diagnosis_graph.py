#!/usr/bin/env python3
"""Build the expert-informed diagnosis graph for the clean expanded train set."""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rca_framework.expert_diagnosis import (
    EXPERT_DIAGNOSIS_GRAPH_VERSION,
    annotation_pattern_audit,
    build_expert_diagnosis_graph,
    review_training_case,
    summarize_training_graphs,
)


DEFAULT_EXPERIMENT = ROOT / "experiments" / "20260816_expanded-pattern-conflict"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_EXPERIMENT / "clean_train.jsonl")
    parser.add_argument("--data-contract", type=Path, default=DEFAULT_EXPERIMENT / "data_contract.json")
    parser.add_argument(
        "--annotations", type=Path,
        default=DEFAULT_EXPERIMENT / "expert_adjudications.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXPERIMENT)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _lanes(case: Mapping[str, Any], metric: str, side: str) -> list[float]:
    value = case.get(metric, {}).get(side, {})
    if not isinstance(value, Mapping):
        return []
    result = []
    for raw in value.values():
        try:
            if raw is not None:
                result.append(float(raw))
        except (TypeError, ValueError):
            continue
    return result


def secondary_feature_audit(
    cases: Sequence[Mapping[str, Any]], graphs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Descriptive comparison only; it does not create a new threshold."""
    grouped: dict[tuple[str, str], list[dict[str, float]]] = {}
    for case, graph in zip(cases, graphs):
        for node in graph["nodes"]:
            node_id = str(node["id"])
            if not node_id.startswith("pattern:EP_RX_HARD_DOWN:"):
                continue
            side = node_id.rsplit(":", 1)[-1]
            peer = "L2" if side == "L1" else "L1"
            values: dict[str, float] = {}
            for metric, endpoint, key in (
                ("txpower", peer, "peer_tx_mean"), ("txpower", side, "local_tx_mean"),
                ("rxpower", peer, "peer_rx_mean"), ("bias", peer, "peer_bias_mean"),
            ):
                lanes = [value for value in _lanes(case, metric, endpoint) if value > -39.0]
                if lanes:
                    values[key] = mean(lanes)
            peer_tx = _lanes(case, "txpower", peer)
            peer_tx_valid = [value for value in peer_tx if value > -39.0]
            if peer_tx_valid:
                values["peer_tx_healthy_lane_spread"] = max(peer_tx_valid) - min(peer_tx_valid)
            if peer_tx:
                values["peer_tx_off_lane_count"] = float(sum(value <= -39.0 for value in peer_tx))
            local_rx = [value for value in _lanes(case, "rxpower", side) if value > -39.0]
            if local_rx:
                values["local_surviving_rx_mean"] = mean(local_rx)
            grouped.setdefault((node_id.removeprefix("pattern:"), str(case.get("label"))), []).append(values)
    rows = []
    for (pattern, label), items in sorted(grouped.items()):
        metrics = {}
        for key in sorted({key for item in items for key in item}):
            values = [item[key] for item in items if key in item]
            metrics[key] = {
                "median": round(median(values), 6), "minimum": round(min(values), 6),
                "maximum": round(max(values), 6), "count": len(values),
            }
        rows.append({"pattern": pattern, "label": label, "support": len(items), "metrics": metrics})
    return rows


def render_html(
    summary: Mapping[str, Any], audit: Mapping[str, Any], cases: Sequence[Mapping[str, Any]],
    graphs: Sequence[Mapping[str, Any]], reviews: Sequence[Mapping[str, Any]], label_status: Mapping[str, str],
) -> str:
    pattern_parts = []
    for row in summary["pattern_summary"]:
        purity = "" if row["majority_purity"] is None else format(row["majority_purity"], ".1%")
        pattern_parts.append(
            "<tr>"
            f"<td><code>{esc(row['pattern_id'])}</code></td><td>{esc(row['title'])}</td>"
            f"<td>{esc(row['status'])}</td><td>{row['support']}</td>"
            f"<td><code>{esc(json.dumps(row['label_distribution'], ensure_ascii=False))}</code></td>"
            f"<td>{purity}</td><td>{esc(row['meaning'])}</td></tr>"
        )
    pattern_rows = "".join(pattern_parts)
    pattern_side_rows = "".join(
        "<tr>"
        f"<td><code>{esc(row['pattern'])}</code></td><td>{row['support']}</td>"
        f"<td><code>{esc(json.dumps(row['by_label_status']['expert_reviewed'], ensure_ascii=False))}</code></td>"
        f"<td><code>{esc(json.dumps(row['by_label_status']['unreviewed'], ensure_ascii=False))}</code></td>"
        f"<td>{row['majority_purity']:.1%}</td></tr>"
        for row in summary["pattern_side_summary"]
    )
    boundary_rows = "".join(
        f"<tr><td><code>{esc(key)}</code></td><td>{esc(value)}</td></tr>"
        for key, value in summary["boundary_semantics"].items()
    )
    unsafe_rows = "".join(
        f"<tr><td>{esc(row['pattern_id'])}</td><td>{esc(row['left_case_id'])} ↔ {esc(row['right_case_id'])}</td>"
        f"<td>{esc(row['notes'])}</td><td>{esc(row['reason'])}</td></tr>"
        for row in audit["requires_domain_confirmation"]
    )
    focus_rows = "".join(
        "<tr>"
        f"<td><code>{esc(row['case_id'])}</code></td><td>{esc(row['label'])}</td>"
        f"<td>{esc(row['label_status'])}</td><td>{esc(row['review_class'])}</td>"
        f"<td>{esc(', '.join(row['candidate_set']))}</td><td>{esc(row['label_assessment'])}</td>"
        f"<td>{esc(row['rationale'])}</td></tr>"
        for row in reviews if row["review_class"] != "insufficient_snapshot" or row["label"] == "fiber"
    )
    case_rows = []
    review_by_id = {str(row["case_id"]): row for row in reviews}
    for case, graph in zip(cases, graphs):
        case_id = str(case["case_id"])
        review = review_by_id[case_id]
        evidence = [
            node for node in graph["nodes"]
            if node["type"] in {"MeasurementState", "BoundaryEvidence", "ContinuousEvidence", "TopologyContext"}
        ]
        candidates = [
            {"src": edge["src"], "type": edge["type"], "dst": edge["dst"], "reason": edge["attrs"].get("reason", "")}
            for edge in graph["edges"] if edge["type"] in {"SUPPORTS", "EXCLUDES", "COMPETES_WITH"}
        ]
        case_rows.append(
            "<details class='case'><summary>"
            f"<code>{esc(case_id)}</code><b>{esc(case.get('label'))}</b>"
            f"<span>{esc(label_status.get(case_id, 'unreviewed'))}</span>"
            f"<span>{esc(review['review_class'])} / {esc(review['label_assessment'])}</span>"
            "</summary>"
            f"<p><b>组合复核：</b>{esc(review['rationale'])}</p>"
            f"<p><b>当前候选：</b>{esc(', '.join(review['candidate_set']))}；<b>还需补采：</b>{esc(', '.join(review['required_evidence']) or '无')}</p>"
            "<div class='case-grid'>"
            f"<section><h4>物理/连续证据</h4><pre>{esc(json.dumps(evidence, ensure_ascii=False, indent=2))}</pre></section>"
            f"<section><h4>候选、排除与竞争关系</h4><pre>{esc(json.dumps(candidates, ensure_ascii=False, indent=2))}</pre></section>"
            "</div>"
            f"<p><b>关键缺失：</b>{esc(', '.join(graph['missing_evidence']) or '无')}</p>"
            f"<details><summary>完整图 JSON</summary><pre>{esc(json.dumps(graph, ensure_ascii=False, indent=2))}</pre></details>"
            f"<details><summary>离线标签复核 JSON</summary><pre>{esc(json.dumps(review, ensure_ascii=False, indent=2))}</pre></details>"
            "</details>"
        )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>训练集专家诊断证据链图 v1</title>
<style>
body{{font-family:ui-sans-serif,system-ui,-apple-system,"PingFang SC",sans-serif;margin:0;background:#f4f7fb;color:#172033}}
main{{max-width:1480px;margin:auto;padding:28px}}h1,h2{{letter-spacing:-.02em}}section,.case{{background:white;border:1px solid #dce3ed;border-radius:12px;padding:18px;margin:14px 0;box-shadow:0 2px 10px #1720330b}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}}.card{{background:#eef5ff;padding:14px;border-radius:10px}}.card b{{display:block;font-size:26px;margin-top:5px}}
table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #e4e9f0;text-align:left;vertical-align:top;padding:9px}}th{{background:#f7f9fc;position:sticky;top:0}}
.warn{{border-left:5px solid #d97706;background:#fffbeb;padding:12px}}code,pre{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}pre{{white-space:pre-wrap;word-break:break-word;max-height:430px;overflow:auto;background:#0f172a;color:#dbeafe;padding:12px;border-radius:8px}}
.case>summary{{display:grid;grid-template-columns:210px 70px 140px 1fr;gap:12px;cursor:pointer;align-items:center}}.case-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}@media(max-width:900px){{.case-grid{{grid-template-columns:1fr}}.case>summary{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>训练集专家诊断证据链图 v1</h1>
<p>本报告把“观测图”升级为“物理边界 → 专家模式 → 候选/排除/竞争关系 → 补采动作”。构图过程不读取标签；标签只在图完成后用于汇总支持度与纯度。</p>
<section><h2>固定诊断链</h2><p><code>Q0 数据质量/blackout → P bias与TX是否出光 → R RX/SNR/SerDes及方向关系 → F fiber候选与现场补采 → D 候选、排除、竞争或拒答</code></p><p>每条case图都保存上述SOPStep与PRECEDES边；未命中的步骤仍保留，防止把“没有检查”误写成“检查正常”。</p></section>
<div class="cards"><div class="card">训练 case<b>{summary['case_count']}</b></div><div class="card">诊断签名<b>{summary['unique_diagnostic_signatures']}</b></div><div class="card">混合标签签名组<b>{summary['mixed_label_signature_groups']}</b></div><div class="card">混合组覆盖 case<b>{summary['cases_in_mixed_label_signatures']}</b></div><div class="card">专家 pair<b>{audit['annotation_count']}</b></div><div class="card">待二次领域确认<b>{audit['requires_domain_confirmation_count']}</b></div></div>
<section><h2>组合证据复核结论</h2><p>RX触底只确定异常方向；只有唯一一侧TX/bias失效时才形成决定性端点证据。其余case保留竞争候选，不强行按训练标签解释。</p><table><thead><tr><th>复核类别</th><th>case数</th></tr></thead><tbody>{''.join(f'<tr><td><code>{esc(k)}</code></td><td>{v}</td></tr>' for k,v in summary['review_class_distribution'].items())}</tbody></table><h3>标签评估</h3><table><thead><tr><th>评估</th><th>case数</th></tr></thead><tbody>{''.join(f'<tr><td><code>{esc(k)}</code></td><td>{v}</td></tr>' for k,v in summary['label_assessment_distribution'].items())}</tbody></table></section>
<section><h2>重点逐case复核：方向异常、TX决定性证据与全部fiber标签</h2><table><thead><tr><th>case</th><th>label</th><th>审核状态</th><th>复核类别</th><th>物理候选</th><th>标签评估</th><th>原因</th></tr></thead><tbody>{focus_rows}</tbody></table></section>
<section><h2>RX硬触底组的另一侧/其他特征对比</h2><p>下列仅是按当前训练标签分组的中位数与范围，不生成新阈值。样本量小且标签审核状态混合，因此只能用于寻找下一步复核线索。</p><pre>{esc(json.dumps(summary['secondary_feature_audit'], ensure_ascii=False, indent=2))}</pre></section>
<section><h2>物理边界语义</h2><p class="warn"><b>零值必须按指标解释：</b>光功率 0 dBm 是正常有光；SNR 0、SerDes ≤1、bias 0 才分别表示触底、失效状态和激光器未驱动。</p><table><thead><tr><th>边界</th><th>语义</th></tr></thead><tbody>{boundary_rows}</tbody></table></section>
<section><h2>专家模式在122条训练集上的回放</h2><table><thead><tr><th>ID</th><th>模式</th><th>权限</th><th>支持数</th><th>标签分布</th><th>多数纯度</th><th>解释</th></tr></thead><tbody>{pattern_rows}</tbody></table></section>
<section><h2>模式方向与标签审核状态</h2><p>同一模式必须按发生侧拆开，并把专家确认标签与沿用旧标签分开。旧标签的多数分布不能覆盖专家确认模式。</p><table><thead><tr><th>模式:侧</th><th>支持数</th><th>专家审核标签</th><th>未审核原标签</th><th>混合口径纯度</th></tr></thead><tbody>{pattern_side_rows}</tbody></table></section>
<section><h2>未自动沉淀的专家说明</h2><p>以下说明依赖跨端 TX-RX 绝对损耗。当前数据不满足lane对应和标定前提，因此仅保留审计记录。</p><table><thead><tr><th>pair</th><th>case</th><th>专家说明</th><th>暂缓原因</th></tr></thead><tbody>{unsafe_rows}</tbody></table></section>
<h2>逐 case 诊断图（122/122）</h2>{''.join(case_rows)}
</main></body></html>"""


def main() -> None:
    args = parse_args()
    cases = load_jsonl(args.train)
    contract = json.loads(args.data_contract.read_text(encoding="utf-8"))
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    label_status = {
        str(row["case_id"]): str(row.get("label_status", "unreviewed"))
        for row in contract.get("cases", ()) if row.get("split") == "train"
    }
    graphs = [build_expert_diagnosis_graph(case) for case in cases]
    summary = summarize_training_graphs(cases, graphs, label_status)
    summary["secondary_feature_audit"] = secondary_feature_audit(cases, graphs)
    audit = annotation_pattern_audit(annotations)
    unsafe_case_ids = {
        str(row[key]) for row in audit["requires_domain_confirmation"]
        for key in ("left_case_id", "right_case_id")
    }
    reviews = [
        review_training_case(
            case, graph, label_status=label_status.get(str(case["case_id"]), "unreviewed"),
            unsafe_expert_reasoning=str(case["case_id"]) in unsafe_case_ids,
        )
        for case, graph in zip(cases, graphs)
    ]
    summary["review_class_distribution"] = dict(Counter(row["review_class"] for row in reviews))
    summary["label_assessment_distribution"] = dict(Counter(row["label_assessment"] for row in reviews))
    summary["suspected_label_conflicts"] = [
        row for row in reviews if row["label_assessment"] == "suspected_label_conflict"
    ]
    payload = {
        "schema_version": EXPERT_DIAGNOSIS_GRAPH_VERSION,
        "source_train": str(args.train), "source_annotations": str(args.annotations),
        "summary": summary, "annotation_audit": audit, "cases": graphs, "offline_label_reviews": reviews,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "expert_diagnosis_graph_v1.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    (args.output_dir / "expert_diagnosis_graph_summary.json").write_text(
        json.dumps({"summary": summary, "annotation_audit": audit}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "expert_diagnosis_graph_analysis.html").write_text(
        render_html(summary, audit, cases, graphs, reviews, label_status), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
