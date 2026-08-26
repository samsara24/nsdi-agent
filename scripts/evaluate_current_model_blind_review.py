#!/usr/bin/env python3
"""Evaluate the frozen current-model blind review and render a standalone HTML report."""

from __future__ import annotations

import hashlib
import html
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/current_model_case_review_v1"
DATA = ROOT / "datasets/filtered_rule_temporal_2025_06_09_v1/test"
PREDICTIONS = ARTIFACT / "model_predictions_draft.json"
FREEZE = ARTIFACT / "blind_prediction_freeze.json"
EXPERT_PREDICTIONS = ARTIFACT / "expert_augmented_predictions.json"
EXPERT_FREEZE = ARTIFACT / "expert_augmented_freeze.json"
LABELS = ("L1", "L2", "fiber")
SPLITS = (("all_data", 417), ("rule1_channel_not_4", 67))


def safe_div(num: int, den: int) -> float:
    return num / den if den else 0.0


def load_and_verify() -> tuple[list[dict], dict]:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    digest = hashlib.sha256(PREDICTIONS.read_bytes()).hexdigest()
    if digest != freeze["sha256"]:
        raise RuntimeError(f"frozen prediction hash mismatch: {digest} != {freeze['sha256']}")
    predictions = json.loads(PREDICTIONS.read_text(encoding="utf-8"))
    if len(predictions) != freeze["case_count"]:
        raise RuntimeError("prediction count no longer matches freeze manifest")
    return predictions, freeze


def evaluate_split(name: str, predictions: list[dict]) -> dict:
    truth = {}
    contracts = {}
    for path in sorted((DATA / name).glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        truth[row["case_id"]] = row["label"]
        contracts[row["case_id"]] = row.get("_dataset_contract", {})
    rows = []
    confusion = {actual: {pred: 0 for pred in LABELS} for actual in LABELS}
    for pred in predictions:
        case_id = pred["case_id"]
        actual = truth[case_id]
        predicted = pred["verdict"]
        confusion[actual][predicted] += 1
        rows.append({
            **pred,
            "dataset": name,
            "actual": actual,
            "correct": predicted == actual,
            "label_status": contracts[case_id].get("label_status", "unknown"),
            "parsed_month": contracts[case_id].get("parsed_month"),
            "original_label": contracts[case_id].get("original_label"),
        })
    class_metrics = {}
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[actual][label] for actual in LABELS if actual != label)
        fn = sum(confusion[label][pred] for pred in LABELS if pred != label)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        class_metrics[label] = {
            "support": sum(confusion[label].values()),
            "predicted": sum(confusion[actual][label] for actual in LABELS),
            "precision": precision,
            "recall": recall,
            "f1": safe_div(2 * precision * recall, precision + recall),
        }
    correct = sum(row["correct"] for row in rows)
    return {
        "dataset": name,
        "case_count": len(rows),
        "correct": correct,
        "accuracy": safe_div(correct, len(rows)),
        "truth_distribution": dict(Counter(row["actual"] for row in rows)),
        "prediction_distribution": dict(Counter(row["verdict"] for row in rows)),
        "confusion_matrix": confusion,
        "class_metrics": class_metrics,
        "rows": rows,
    }


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def render_report(results: list[dict], expert_results: list[dict], freeze: dict, expert_freeze: dict) -> str:
    total = sum(r["case_count"] for r in results)
    correct = sum(r["correct"] for r in results)
    cards = "".join(
        f'<div class="card"><h3>{html.escape(cold["dataset"])} · 冷启动</h3>'
        f'<div class="score">{pct(cold["accuracy"])}</div><p>{cold["correct"]} / {cold["case_count"]} correct</p></div>'
        f'<div class="card expert"><h3>{html.escape(expert["dataset"])} · 专家增强</h3>'
        f'<div class="score">{pct(expert["accuracy"])}</div><p>{expert["correct"]} / {expert["case_count"]} correct；变化 {expert["correct"]-cold["correct"]:+d}</p></div>'
        for cold, expert in zip(results, expert_results)
    )
    diagnostic_items = []
    for result, expert_result in zip(results, expert_results):
        majority = max(result["truth_distribution"].values())
        majority_accuracy = safe_div(majority, result["case_count"])
        unreviewed = sum(row["label_status"] == "unreviewed" for row in result["rows"])
        fiber = result["class_metrics"]["fiber"]
        l2 = result["class_metrics"]["L2"]
        diagnostic_items.append(
            f"<li><strong>{html.escape(result['dataset'])}</strong>：盲审 accuracy {pct(result['accuracy'])}，"
            f"低于该测试集多数类基线 {pct(majority_accuracy)}；"
            f"fiber 真实 {fiber['support']} 条、预测 {fiber['predicted']} 条、precision {pct(fiber['precision'])}；"
            f"L2 recall {pct(l2['recall'])}；专家增强后 accuracy {pct(expert_result['accuracy'])}（{expert_result['correct']-result['correct']:+d} 条）；"
            f"{unreviewed}/{result['case_count']} 条权威文件仍标记为 unreviewed。</li>"
        )
    diagnostics = f"""
    <section><h2>结果复盘与风险判断</h2>
      <ul>{''.join(diagnostic_items)}</ul>
      <p><strong>逐 case 变化：</strong>{'; '.join(
          f'{cold["dataset"]} 救回 {sum((not c["correct"]) and e["correct"] for c, e in zip(cold["rows"], expert["rows"]))} 条、'
          f'干扰 {sum(c["correct"] and (not e["correct"]) for c, e in zip(cold["rows"], expert["rows"]))} 条'
          for cold, expert in zip(results, expert_results)
      )}。这说明专家知识带来净提升，但不是对所有 case 单调改进。</p>
      <p><strong>主要模型偏差：</strong>当前冷启动判断把“单 lane 接收功率低于对端发送、LOS/LOL 或 SNR 下降”过度解释成 fiber，
      但现有人工标签大量仍归在端点类，导致 fiber 严重过预测。400G–400G 数据中，告警接口统一显示为 L1，缺少 SerDes 与 lane 编号补充信息时，
      冷启动模型又过度依赖告警端，因而真实 L2 大量被判成 L1。</p>
      <p><strong>标签风险：</strong>标签状态为 unreviewed 不等于标签错误，但意味着这些差异不能直接全部归因于模型。
      尤其是“对端发送有效、同号 lane 接收严重衰减或中断”却被标成端点类的 case，应进入专家复核；在复核完成前，报告同时保留冻结预测和现有权威标签，不反向修改判断。</p>
      <p><strong>流程结论：</strong>仅靠冷启动通用物理直觉无法达到 80%–90%。后续知识流程至少需要从训练集校准：
      端点标签与链路衰减证据的实际标注边界、400G–400G 两端方向先验，以及缺少 SerDes 时哪些光学异常允许判 fiber。</p>
      <p><strong>专家规则局限：</strong>文档规则把无异常 case 固定返回 L1，并且只有“两端最高规则同优先级且定界相反”才返回 fiber。
      因此它修复了冷启动 fiber 过预测和 400G–400G 的 L2 完全失认问题，但同时形成明显 L1 偏置，并把大量真实 fiber 回收到端点类别。</p>
    </section>"""
    sections = []
    all_rows = []
    for result, expert_result in zip(results, expert_results):
        cm_head = "".join(f"<th>预测 {x}</th>" for x in LABELS)
        cm_rows = "".join(
            f'<tr><th>真实 {actual}</th>' + "".join(
                f'<td>{result["confusion_matrix"][actual][pred]}</td>' for pred in LABELS
            ) + "</tr>" for actual in LABELS
        )
        metric_rows = "".join(
            f'<tr><td>{label}</td><td>{m["support"]}</td><td>{m["predicted"]}</td>'
            f'<td>{pct(m["precision"])}</td><td>{pct(m["recall"])}</td><td>{pct(m["f1"])}</td></tr>'
            for label, m in result["class_metrics"].items()
        )
        expert_cm_rows = "".join(
            f'<tr><th>真实 {actual}</th>' + "".join(
                f'<td>{expert_result["confusion_matrix"][actual][pred]}</td>' for pred in LABELS
            ) + "</tr>" for actual in LABELS
        )
        expert_metric_rows = "".join(
            f'<tr><td>{label}</td><td>{m["support"]}</td><td>{m["predicted"]}</td>'
            f'<td>{pct(m["precision"])}</td><td>{pct(m["recall"])}</td><td>{pct(m["f1"])}</td></tr>'
            for label, m in expert_result["class_metrics"].items()
        )
        sections.append(f"""
        <section>
          <h2>{html.escape(result['dataset'])}</h2>
          <p>冷启动 <strong>{pct(result['accuracy'])}</strong> ({result['correct']}/{result['case_count']})；专家增强 <strong>{pct(expert_result['accuracy'])}</strong> ({expert_result['correct']}/{expert_result['case_count']})。</p>
          <div class="grid"><table><caption>冷启动混淆矩阵</caption><thead><tr><th></th>{cm_head}</tr></thead><tbody>{cm_rows}</tbody></table>
          <table><caption>专家增强混淆矩阵</caption><thead><tr><th></th>{cm_head}</tr></thead><tbody>{expert_cm_rows}</tbody></table>
          <table><caption>冷启动分类指标</caption><thead><tr><th>标签</th><th>真实数</th><th>预测数</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>{metric_rows}</tbody></table>
          <table><caption>专家增强分类指标</caption><thead><tr><th>标签</th><th>真实数</th><th>预测数</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>{expert_metric_rows}</tbody></table></div>
          <p>真实分布：{html.escape(json.dumps(result['truth_distribution'], ensure_ascii=False))}<br>
             预测分布：{html.escape(json.dumps(result['prediction_distribution'], ensure_ascii=False))}</p>
        </section>""")
        all_rows.extend(result["rows"])
    expert_by_case = {row["case_id"]: row for result in expert_results for row in result["rows"]}
    case_rows = "".join(
        f'<tr class="{"ok" if expert_by_case[row["case_id"]]["correct"] else "bad"}" data-dataset="{html.escape(row["dataset"])}" data-result="{"correct" if expert_by_case[row["case_id"]]["correct"] else "wrong"}">'
        f'<td>{html.escape(row["dataset"])}</td><td><code>{html.escape(row["case_id"])}</code></td>'
        f'<td>{html.escape(row["actual"])}</td><td>{html.escape(row["verdict"])}</td><td>{"正确" if row["correct"] else "错误"}</td>'
        f'<td>{html.escape(expert_by_case[row["case_id"]]["verdict"])}</td><td>{"正确" if expert_by_case[row["case_id"]]["correct"] else "错误"}</td>'
        f'<td><code>{html.escape(expert_by_case[row["case_id"]]["rule"])}</code><br>{html.escape(expert_by_case[row["case_id"]]["reasoning"])}</td>'
        f'<td>{html.escape(row["reasoning"])}</td><td>{html.escape(str(row["label_status"]))}</td></tr>'
        for row in all_rows
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>当前模型逐 Case 盲审复盘</title><style>
    :root{{--bg:#f5f7fb;--ink:#18212f;--muted:#667085;--line:#d8dee9;--good:#e8f7ee;--bad:#fff0f0;--accent:#315efb}}
    body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
    main{{max-width:1500px;margin:auto;padding:28px}} h1{{margin-bottom:6px}} .muted{{color:var(--muted)}}
    .cards,.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}}
    .card,section{{background:white;border:1px solid var(--line);border-radius:12px;padding:18px;margin:16px 0;box-shadow:0 2px 8px #18212f0a}}
    .score{{font-size:34px;font-weight:750;color:var(--accent)}} table{{width:100%;border-collapse:collapse;background:white}}
    caption{{text-align:left;font-weight:700;margin:8px 0}} th,td{{border:1px solid var(--line);padding:8px;vertical-align:top}} th{{background:#f1f4f9}}
    .cases{{font-size:12px}} .cases tr.ok td{{background:var(--good)}} .cases tr.bad td{{background:var(--bad)}} code{{white-space:nowrap}}
    .toolbar{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}} select{{padding:7px;border:1px solid var(--line);border-radius:7px}}
    </style></head><body><main><h1>当前模型逐 Case 盲审复盘</h1>
    <p class="muted">冷启动预测 SHA-256：<code>{freeze['sha256']}</code>；专家增强预测 SHA-256：<code>{expert_freeze['sha256']}</code>。专家规则固定后仅读取去标签遥测，未按测试标签调参。</p>
    <div class="cards">{cards}</div>
    {diagnostics}{''.join(sections)}
    <section><h2>逐 Case 对照</h2><div class="toolbar"><label>数据集 <select id="dataset"><option value="all">全部</option><option>all_data</option><option>rule1_channel_not_4</option></select></label><label>结果 <select id="result"><option value="all">全部</option><option value="wrong">仅错误</option><option value="correct">仅正确</option></select></label></div>
    <table class="cases"><thead><tr><th>数据集</th><th>Case</th><th>真实</th><th>冷启动</th><th>冷启动结果</th><th>专家增强</th><th>增强结果</th><th>专家规则与分析</th><th>冷启动理由</th><th>标签状态</th></tr></thead><tbody>{case_rows}</tbody></table></section>
    <script>const ds=document.querySelector('#dataset'),rs=document.querySelector('#result');function f(){{document.querySelectorAll('.cases tbody tr').forEach(r=>r.hidden=!((ds.value==='all'||r.dataset.dataset===ds.value)&&(rs.value==='all'||r.dataset.result===rs.value)))}}ds.onchange=f;rs.onchange=f;</script>
    </main></body></html>"""


def main() -> None:
    predictions, freeze = load_and_verify()
    expert_freeze = json.loads(EXPERT_FREEZE.read_text(encoding="utf-8"))
    expert_digest = hashlib.sha256(EXPERT_PREDICTIONS.read_bytes()).hexdigest()
    if expert_digest != expert_freeze["sha256"]:
        raise RuntimeError("expert prediction hash mismatch")
    expert_predictions = json.loads(EXPERT_PREDICTIONS.read_text(encoding="utf-8"))
    offset = 0
    results = []
    expert_results = []
    for name, expected in SPLITS:
        part = predictions[offset:offset + expected]
        result = evaluate_split(name, part)
        if result["case_count"] != expected:
            raise RuntimeError(f"unexpected case count for {name}")
        results.append(result)
        expert_results.append(evaluate_split(name, expert_predictions[offset:offset + expected]))
        offset += expected
    payload = {"freeze": freeze, "expert_freeze": expert_freeze, "datasets": results, "expert_datasets": expert_results}
    (ARTIFACT / "blind_evaluation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ARTIFACT / "blind_review_report.html").write_text(render_report(results, expert_results, freeze, expert_freeze), encoding="utf-8")
    for result in results:
        print(result["dataset"], result["correct"], result["case_count"], f"{result['accuracy']:.6f}")
    for result in expert_results:
        print("expert", result["dataset"], result["correct"], result["case_count"], f"{result['accuracy']:.6f}")


if __name__ == "__main__":
    main()
