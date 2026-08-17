"""Render self-contained offline HTML reports for routing experiments.

The renderer intentionally depends only on the Python standard library.  It
accepts the JSON-compatible objects written by ``scripts/evaluate_routing.py``
and tolerates partially populated records so older experiment artifacts can
still be inspected.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


REPORT_SCHEMA = "rca-experiment-html-v1"
_ROOT_CAUSES = ("L1", "L2", "fiber")
_CLASS_LABELS = {
    "correct": "回答正确",
    "wrong": "模型答错",
    "degraded_correct": "候选正确但被 M9 降级",
    "degraded_wrong": "候选错误且被 M9 降级",
    "abstain": "弃权/补采",
    "telemetry": "遥测不足",
}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _get(value: Any, *keys: str, default: Any = None) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, Mapping):
            return default
        current = current.get(key, default)
        if current is default:
            return default
    return current


def _text(value: Any) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return str(value)


def _esc(value: Any) -> str:
    return html.escape(_text(value), quote=True)


def _json_block(value: Any) -> str:
    if value is None:
        return '<p class="empty">无数据</p>'
    try:
        rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    except (TypeError, ValueError):
        rendered = str(value)
    return f"<pre>{html.escape(rendered, quote=True)}</pre>"


def _number(value: Any, digits: int = 2) -> str:
    if isinstance(value, bool):
        return _text(value)
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return _text(value)


def _percent(value: Any) -> str:
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return _text(value)


def _slug(value: Any, fallback: str) -> str:
    raw = str(value or fallback)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip(".-") or fallback
    if slug != raw:
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
        slug = f"{slug}-{digest}"
    return slug[:160]


def _dedupe(values: Iterable[Any]) -> List[Any]:
    result: List[Any] = []
    seen: set[str] = set()
    for value in values:
        marker = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        if marker not in seen:
            seen.add(marker)
            result.append(value)
    return result


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]], classes: str = "") -> str:
    if not rows:
        return '<p class="empty">无数据</p>'
    head = "".join(f"<th>{_esc(item)}</th>" for item in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_esc(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    class_attr = f' class="{html.escape(classes, quote=True)}"' if classes else ""
    return f"<div class=\"table-wrap\"><table{class_attr}><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _kv_table(items: Sequence[Tuple[str, Any]]) -> str:
    rows = [
        f"<tr><th>{_esc(key)}</th><td>{_esc(value)}</td></tr>"
        for key, value in items
    ]
    return '<div class="table-wrap"><table class="kv"><tbody>' + "".join(rows) + "</tbody></table></div>"


def _panel(title: str, body: str, *, panel_id: str = "", open_panel: bool = True) -> str:
    id_attr = f' id="{html.escape(panel_id, quote=True)}"' if panel_id else ""
    open_attr = " open" if open_panel else ""
    return (
        f"<details class=\"panel\"{id_attr}{open_attr}>"
        f"<summary>{_esc(title)}</summary><div class=\"panel-body\">{body}</div></details>"
    )


def _document(title: str, body: str, *, index: bool = False) -> str:
    script = ""
    if index:
        script = """
<script>
(() => {
  const input = document.getElementById("case-filter");
  if (!input) return;
  input.addEventListener("input", () => {
    const query = input.value.toLocaleLowerCase();
    document.querySelectorAll("[data-case-row]").forEach((row) => {
      row.hidden = !row.textContent.toLocaleLowerCase().includes(query);
    });
  });
})();
</script>"""
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>
:root{{--bg:#f4f6f8;--card:#fff;--ink:#18212b;--muted:#66717e;--line:#dce2e8;
--blue:#2458a6;--green:#147d55;--red:#b42318;--amber:#a15c00;--purple:#6b47a8}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.55 system-ui,-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}}
main{{max-width:1440px;margin:auto;padding:24px}} h1{{font-size:26px;margin:0 0 6px}}
h2{{font-size:19px;margin:26px 0 10px}} h3{{font-size:16px;margin:18px 0 8px}}
a{{color:var(--blue);text-decoration:none}} a:hover{{text-decoration:underline}}
.subtitle,.muted,.empty{{color:var(--muted)}} .empty{{font-style:italic}}
.toolbar{{display:flex;gap:12px;align-items:center;margin:18px 0}}
input[type=search]{{width:min(480px,100%);padding:10px 12px;border:1px solid var(--line);
border-radius:7px;background:white}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:10px;margin:12px 0 20px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:13px}}
.card .value{{font-size:22px;font-weight:700;margin-top:3px}} .card .label{{color:var(--muted)}}
.panel{{background:var(--card);border:1px solid var(--line);border-radius:9px;margin:10px 0;overflow:hidden}}
.panel>summary{{cursor:pointer;font-size:16px;font-weight:650;padding:12px 15px;background:#fafbfc}}
.panel-body{{padding:14px 15px;border-top:1px solid var(--line)}}
.table-wrap{{overflow:auto}} table{{border-collapse:collapse;width:100%;background:white}}
th,td{{padding:8px 10px;border:1px solid var(--line);vertical-align:top;text-align:left}}
thead th,.kv th{{background:#f5f7f9;font-weight:650}} .kv th{{width:220px}}
.bar{{height:10px;background:#e5e7eb;border-radius:999px;overflow:hidden;min-width:120px}}
.bar span{{display:block;height:100%;background:var(--blue)}}
pre{{margin:8px 0;padding:12px;max-height:520px;overflow:auto;white-space:pre-wrap;word-break:break-word;
background:#111827;color:#e5edf6;border-radius:7px;font:12px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace}}
code,.token{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}}
.badge{{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;font-weight:650;white-space:nowrap}}
.correct{{color:var(--green);background:#e8f5ef}} .wrong{{color:var(--red);background:#fdecea}}
.degraded_correct{{color:#176b47;background:#edf8f2}} .degraded_wrong{{color:#9b2c20;background:#fff1ef}}
.abstain{{color:var(--amber);background:#fff4df}} .telemetry{{color:var(--purple);background:#f1ebfb}}
.tokens{{display:flex;flex-wrap:wrap;gap:6px}} .token{{padding:3px 7px;border:1px solid var(--line);border-radius:5px;background:#f7f9fb}}
.attempt{{border-left:4px solid var(--blue);padding-left:12px;margin:18px 0}}
.violation{{border-left:4px solid var(--red);padding:7px 10px;background:#fff6f5;margin:6px 0}}
.ok{{color:var(--green);font-weight:650}} .warn{{color:var(--amber);font-weight:650}}
.crumb{{margin-bottom:14px}} .group{{margin:18px 0 28px}} [hidden]{{display:none!important}}
@media print{{body{{background:white}} main{{max-width:none;padding:0}} .toolbar{{display:none}}
.panel{{break-inside:avoid}} details.panel:not([open])>.panel-body{{display:block}}}}
</style>
</head>
<body><main>{body}</main>{script}</body>
</html>
"""


def _final_decision(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(record.get("final_decision"))


def _branch_outcome(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(record.get("branch_outcome"))


def _routing(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(record.get("routing"))


def _actual(record: Mapping[str, Any]) -> Any:
    return record.get("actual")


def _prediction(record: Mapping[str, Any]) -> Any:
    decision = _final_decision(record)
    if decision.get("verdict") not in (None, ""):
        return decision.get("verdict")
    return None


def _proposed_prediction(record: Mapping[str, Any]) -> Any:
    decision = _final_decision(record)
    return decision.get("proposed_verdict") or _branch_outcome(record).get("verdict")


def _telemetry_status(record: Mapping[str, Any]) -> str:
    features = _mapping(record.get("features"))
    evidence_pack = _mapping(record.get("evidence_pack"))
    status = features.get("telemetry_status") or evidence_pack.get("telemetry_status")
    if status:
        return str(status)
    return ""


def _is_telemetry_insufficient(record: Mapping[str, Any]) -> bool:
    status = _telemetry_status(record).lower()
    if status in {"no_telemetry", "invalid_telemetry", "telemetry_blackout", "unavailable"}:
        return True
    routing = _routing(record)
    outcome = _branch_outcome(record)
    branch = str(routing.get("branch") or outcome.get("branch") or "")
    if branch == "N6":
        return True
    searchable = " ".join(
        str(value)
        for value in (
            routing.get("reason"),
            outcome.get("caveats"),
            _final_decision(record).get("reason"),
        )
        if value
    ).lower()
    return any(
        phrase in searchable
        for phrase in (
            "no telemetry",
            "telemetry unavailable",
            "optical blackout",
            "遥测不足",
            "遥测失效",
            "无遥测",
            "全链路失效",
        )
    )


def _classification(record: Mapping[str, Any]) -> str:
    actual = _actual(record)
    prediction = _prediction(record)
    if prediction is not None:
        return "correct" if prediction == actual else "wrong"
    proposed = _proposed_prediction(record)
    if proposed is not None:
        return "degraded_correct" if proposed == actual else "degraded_wrong"
    if _is_telemetry_insufficient(record):
        return "telemetry"
    return "abstain"


def _policy_summary(summary: Mapping[str, Any], policy: str) -> Mapping[str, Any]:
    policies = _mapping(summary.get("policies"))
    if policy in policies:
        return _mapping(policies.get(policy))
    return _mapping(summary.get(policy))


def _policy_stats(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    classes = Counter(_classification(record) for record in records)
    routing = Counter(
        str(_routing(record).get("branch") or _branch_outcome(record).get("branch") or "unknown")
        for record in records
    )
    answered = classes["correct"] + classes["wrong"]
    proposed = answered + classes["degraded_correct"] + classes["degraded_wrong"]
    return {
        "total": len(records),
        "answered": answered,
        "correct": classes["correct"],
        "wrong": classes["wrong"],
        "degraded_correct": classes["degraded_correct"],
        "degraded_wrong": classes["degraded_wrong"],
        "abstain": classes["abstain"],
        "telemetry": classes["telemetry"],
        "coverage": answered / len(records) if records else 0.0,
        "precision": classes["correct"] / answered if answered else None,
        "proposed": proposed,
        "proposed_correct": classes["correct"] + classes["degraded_correct"],
        "routing": dict(sorted(routing.items())),
    }


def _metric_cards(policy_summary: Mapping[str, Any], stats: Mapping[str, Any]) -> str:
    final = _mapping(policy_summary.get("final_decisions"))
    answered = final.get("answered", stats["answered"])
    correct = final.get("correct", stats["correct"])
    coverage = final.get("coverage", stats["coverage"])
    precision = final.get("precision_when_answered", stats["precision"])
    actions = _mapping(final.get("actions"))
    cards = (
        ("Case 总数", stats["total"]),
        ("最终回答", answered),
        ("答对", correct),
        ("覆盖率", _percent(coverage)),
        ("回答准确率", _percent(precision)),
        ("补采", actions.get("request_evidence", stats["abstain"])),
        ("人工介入", actions.get("human_review", stats["telemetry"])),
        ("降级前候选", stats["proposed"]),
        ("候选正确", stats["proposed_correct"]),
    )
    return '<div class="cards">' + "".join(
        f'<div class="card"><div class="label">{_esc(label)}</div><div class="value">{_esc(value)}</div></div>'
        for label, value in cards
    ) + "</div>"


def _class_metrics(policy_summary: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> str:
    metrics = _mapping(policy_summary.get("forced_class_metrics")) or _mapping(
        _get(policy_summary, "final_decisions", "class_metrics", default={})
    )
    if not metrics:
        labels = list(_ROOT_CAUSES)
        labels.extend(
            sorted(
                {
                    str(value)
                    for record in records
                    for value in (_actual(record), _prediction(record))
                    if value not in (None, "") and value not in labels
                }
            )
        )
        rebuilt: Dict[str, Any] = {}
        for label in labels:
            tp = sum(_actual(record) == label and _prediction(record) == label for record in records)
            fp = sum(_actual(record) != label and _prediction(record) == label for record in records)
            support = sum(_actual(record) == label for record in records)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / support if support else 0.0
            rebuilt[label] = {
                "support": support,
                "predicted": tp + fp,
                "true_positive": tp,
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            }
        metrics = rebuilt
    rows = []
    for label, value in metrics.items():
        row = _mapping(value)
        rows.append(
            (
                label,
                row.get("support"),
                row.get("predicted"),
                row.get("true_positive"),
                _percent(row.get("precision")),
                _percent(row.get("recall")),
                _percent(row.get("f1")),
            )
        )
    return _table(("类别", "真值数", "预测数", "TP", "Precision", "Recall", "F1"), rows)


def _confidence_reliability(policy_summary: Mapping[str, Any]) -> str:
    rows = []
    for item in _sequence(policy_summary.get("confidence_reliability")):
        row = _mapping(item)
        rows.append((
            row.get("bucket"),
            row.get("n"),
            row.get("correct"),
            _percent(row.get("accuracy")),
            _percent(row.get("mean_confidence")),
            _text(row.get("prediction_distribution")),
        ))
    return _table(("置信度桶", "n", "答对", "准确率", "平均置信度", "预测分布"), rows)


def _threshold_sweep(policy_summary: Mapping[str, Any]) -> str:
    rows = []
    for item in _sequence(policy_summary.get("threshold_sweep")):
        row = _mapping(item)
        rows.append((
            _percent(row.get("threshold")),
            row.get("answered"),
            row.get("degraded"),
            _percent(row.get("coverage")),
            row.get("correct"),
            _percent(row.get("precision_when_answered")),
        ))
    return _table(("阈值", "自动结案", "降级", "覆盖率", "答对", "自动结案准确率"), rows)


def _confidence_breakdown_panel(record: Mapping[str, Any]) -> str:
    decision = _final_decision(record)
    outcome = _branch_outcome(record)
    breakdown = _mapping(decision.get("confidence_breakdown")) or _mapping(outcome.get("confidence_breakdown"))
    if not breakdown:
        return '<p class="empty">未记录四维置信度</p>'
    rows = []
    for key in ("evidence_completeness", "physical_compliance", "reasoning_completeness", "history_similarity"):
        value = float(breakdown.get(key, 0.0) or 0.0)
        rows.append(
            "<tr>"
            f"<td>{_esc(key)}</td>"
            f"<td>{_esc(_percent(value))}</td>"
            f'<td><div class="bar"><span style="width:{max(0.0, min(1.0, value)) * 100:.1f}%"></span></div></td>'
            "</tr>"
        )
    penalties = decision.get("compliance_penalties") or outcome.get("compliance_penalties")
    body = (
        '<div class="table-wrap"><table><thead><tr><th>维度</th><th>分数</th><th>条形</th></tr></thead>'
        "<tbody>" + "".join(rows) + "</tbody></table></div>"
        + "<h3>约束扣分明细</h3>" + _json_block(penalties)
    )
    return body


def _routing_table(policy_summary: Mapping[str, Any], stats: Mapping[str, Any]) -> str:
    counts = _mapping(_get(policy_summary, "routing", "counts", default={})) or _mapping(stats.get("routing"))
    branch_rows = _mapping(policy_summary.get("branches"))
    rows = []
    names = list(counts)
    names.extend(name for name in branch_rows if name not in names)
    for name in names:
        branch = _mapping(branch_rows.get(name))
        rows.append(
            (
                name,
                counts.get(name, branch.get("n", 0)),
                branch.get("answered"),
                branch.get("correct"),
                _percent(branch.get("precision_when_answered")),
                branch.get("needs_llm"),
                branch.get("needs_human"),
            )
        )
    return _table(("路由/分支", "数量", "回答", "答对", "回答准确率", "需 LLM", "需人工"), rows)


def _deep_analysis(
    records: Sequence[Mapping[str, Any]],
    policy_summary: Mapping[str, Any],
    traces: Mapping[str, Any],
) -> str:
    """Render an evidence-backed experiment diagnosis, not just metric cards."""

    proposed = [record for record in records if _proposed_prediction(record) is not None]
    proposed_correct = sum(
        _proposed_prediction(record) == _actual(record) for record in proposed
    )
    sop_rows = []
    sop_class_results: Dict[str, Tuple[int, int]] = {}
    sop_total = 0
    sop_correct = 0
    for label in _ROOT_CAUSES:
        matching = [record for record in records if _actual(record) == label]
        available = [
            record for record in matching if _mapping(record.get("sop_prediction"))
        ]
        correct = sum(
            (
                _mapping(record.get("sop_prediction")).get("verdict")
                or _mapping(record.get("sop_prediction")).get("prediction")
            )
            == label
            for record in available
        )
        sop_total += len(available)
        sop_correct += correct
        sop_class_results[label] = (len(available), correct)
        sop_rows.append(
            (label, len(matching), len(available), correct, _percent(correct / len(available) if available else None))
        )

    branch_counts: Dict[str, Counter[str]] = {}
    for record in records:
        branch = str(_routing(record).get("branch") or _branch_outcome(record).get("branch") or "unknown")
        stats = branch_counts.setdefault(branch, Counter())
        candidate = _proposed_prediction(record)
        stats["cases"] += 1
        stats["proposed"] += candidate is not None
        stats["correct"] += candidate is not None and candidate == _actual(record)
    branch_rows = [
        (
            branch,
            values["cases"],
            values["proposed"],
            values["correct"],
            _percent(values["correct"] / values["proposed"] if values["proposed"] else None),
        )
        for branch, values in sorted(branch_counts.items())
    ]

    trace_stats: Counter[str] = Counter()
    violation_kinds: Counter[str] = Counter()
    violation_messages: Counter[str] = Counter()
    for raw_trace in traces.values():
        trace = _mapping(raw_trace)
        trace_stats["traces"] += 1
        trace_stats["rewritten"] += bool(trace.get("rewrote"))
        accepted = _mapping(trace.get("accepted"))
        trace_stats["accepted"] += bool(accepted)
        trace_stats["accepted_verdict"] += accepted.get("verdict") not in (None, "", "abstain")
        for raw_attempt in _sequence(trace.get("attempts")):
            check = _mapping(_mapping(raw_attempt).get("check"))
            for raw_violation in _sequence(check.get("violations")):
                violation = _mapping(raw_violation)
                violation_kinds[str(violation.get("kind") or "unknown")] += 1
                violation_messages[str(violation.get("message") or "unknown")] += 1

    final = _mapping(policy_summary.get("final_decisions"))
    actions = _mapping(final.get("actions"))
    conclusions = []
    if records and final.get("answered", 0) == 0 and proposed:
        conclusions.append(
            f"M9 将 {len(proposed)} 个降级前候选全部拦截；自动最终覆盖率为 0。"
            "这说明当前 train-LOO 置信标定没有任何分组同时达到决策阈值与最小支持数，"
            "本轮结果不能作为无人值守诊断系统上线。"
        )
    if sop_total:
        sop_accuracy = sop_correct / sop_total
        candidate_accuracy = proposed_correct / len(proposed) if proposed else 0.0
        if sop_accuracy > candidate_accuracy:
            conclusions.append(
                f"训练集归纳 SOP 单独命中 {sop_correct}/{sop_total}（{sop_accuracy:.2%}），"
                f"高于分支候选的 {proposed_correct}/{len(proposed)}（{candidate_accuracy:.2%}）。"
                "当前 LLM 仲裁没有把统计先验稳定转化为更高质量的物理结论。"
            )
        unsupported_sop_labels = [
            label
            for label, (available, correct) in sop_class_results.items()
            if available and correct == 0
        ]
        if unsupported_sop_labels:
            conclusions.append(
                f"SOP 对类别 {'、'.join(unsupported_sop_labels)} 的命中为 0；"
                "总体准确率受多数类主导，不能把该总体数值解释为三类均有效。"
            )
    n5c = branch_counts.get("N5c", Counter())
    if n5c.get("proposed"):
        conclusions.append(
            f"冷启动 N5c 形成 {n5c['proposed']} 个候选，仅 {n5c['correct']} 个正确；"
            "它是当前主要质量瓶颈，不能用放宽 M7 校验来掩盖。"
        )
    if trace_stats["traces"]:
        conclusions.append(
            f"测试侧触发 {trace_stats['traces']} 条 LLM 轨迹，"
            f"{trace_stats['accepted']} 条通过约束校验，其中 {trace_stats['accepted_verdict']} 条形成结论；"
            f"累计记录 {sum(violation_kinds.values())} 条违规。"
        )

    conclusion_html = "<ul>" + "".join(f"<li>{_esc(item)}</li>" for item in conclusions) + "</ul>"
    overview = _kv_table(
        (
            ("降级前候选 / 正确", f"{len(proposed)} / {proposed_correct}"),
            ("M9 最终回答", final.get("answered")),
            ("补采 / 人工", f"{actions.get('request_evidence', 0)} / {actions.get('human_review', 0)}"),
            ("LLM 轨迹 / 重写", f"{trace_stats['traces']} / {trace_stats['rewritten']}"),
            ("LLM 通过 / 形成结论", f"{trace_stats['accepted']} / {trace_stats['accepted_verdict']}"),
            ("约束违规总数", sum(violation_kinds.values())),
        )
    )
    violation_rows = [
        (kind, count) for kind, count in violation_kinds.most_common()
    ]
    violation_message_rows = [
        (message, count) for message, count in violation_messages.most_common(8)
    ]
    return (
        "<h3>结论</h3>"
        + conclusion_html
        + "<h3>关键门禁</h3>"
        + overview
        + "<h3>候选质量（M9 降级前）</h3>"
        + _table(("分支", "Case", "候选", "候选正确", "候选准确率"), branch_rows)
        + "<h3>SOP 对照</h3>"
        + _table(("真值", "Case", "SOP 可用", "SOP 正确", "SOP 准确率"), sop_rows)
        + "<h3>LLM 校验失败构成</h3>"
        + _table(("违规类型", "次数"), violation_rows)
        + "<h3>LLM 高频失败原因</h3>"
        + _table(("校验消息", "次数"), violation_message_rows)
    )


def _case_filename(policy: str, case_id: str, used: set[str]) -> str:
    stem = f"{_slug(policy, 'policy')}-{_slug(case_id, 'case')}"
    candidate = f"{stem}.html"
    if candidate not in used:
        used.add(candidate)
        return candidate
    index = 2
    while f"{stem}-{index}.html" in used:
        index += 1
    candidate = f"{stem}-{index}.html"
    used.add(candidate)
    return candidate


def _case_link(filename: str, case_id: Any) -> str:
    return f'<a href="cases/{html.escape(filename, quote=True)}">{_esc(case_id)}</a>'


def _index_page(
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
    outcomes: Mapping[str, Any],
    traces: Mapping[str, Any],
    case_files: Mapping[Tuple[str, int], str],
    training_summary: Any,
) -> str:
    sections: List[str] = [
        "<h1>RCA 离线实验报告</h1>",
        '<p class="subtitle">选择性诊断、路由分布与逐 case 审计</p>',
    ]
    versions = _mapping(manifest.get("versions"))
    data = _mapping(manifest.get("data"))
    sections.append(
        _panel(
            "实验清单",
            _kv_table(
                (
                    ("Schema", manifest.get("schema_version")),
                    ("创建时间", manifest.get("created_at_utc")),
                    ("数据目录", data.get("data_dir")),
                    ("训练 / 测试", f"{_text(data.get('train_size'))} / {_text(data.get('test_size'))}"),
                    ("证据图", versions.get("evidence_graph")),
                    ("特征字典", versions.get("feature_dictionary")),
                    ("约束库", versions.get("constraint_library")),
                    ("SOP", versions.get("sop")),
                    ("Prompt", versions.get("prompt_template")),
                    ("决策策略", versions.get("decision_policy")),
                )
            ),
            open_panel=False,
        )
    )
    if training_summary is not None:
        sections.append(_panel("训练摘要", _json_block(training_summary), open_panel=False))
    sections.append(
        '<div class="toolbar"><label for="case-filter">筛选 case：</label>'
        '<input id="case-filter" type="search" placeholder="输入 policy、case ID、真值、预测、分支或状态"></div>'
    )

    for policy, raw_records in outcomes.items():
        records = [_mapping(item) for item in _sequence(raw_records)]
        policy_summary = _policy_summary(summary, str(policy))
        stats = _policy_stats(records)
        policy_traces = _mapping(traces.get(policy))
        sections.extend(
            (
                f"<h2>Policy：{_esc(policy)}</h2>",
                _metric_cards(policy_summary, stats),
                _panel(
                    "实验深度分析",
                    _deep_analysis(records, policy_summary, policy_traces),
                ),
                _panel("分类指标", _class_metrics(policy_summary, records)),
                _panel("置信度可靠性", _confidence_reliability(policy_summary), open_panel=False),
                _panel("阈值扫描", _threshold_sweep(policy_summary), open_panel=False),
                _panel("路由分布", _routing_table(policy_summary, stats)),
            )
        )
        grouped: Dict[str, List[Tuple[int, Mapping[str, Any]]]] = {
            key: [] for key in _CLASS_LABELS
        }
        for index, record in enumerate(records):
            grouped[_classification(record)].append((index, record))
        for classification in (
            "wrong",
            "degraded_wrong",
            "abstain",
            "telemetry",
            "degraded_correct",
            "correct",
        ):
            items = grouped[classification]
            label = _CLASS_LABELS[classification]
            rows: List[str] = []
            for index, record in items:
                case_id = record.get("case_id", f"case-{index + 1}")
                filename = case_files[(str(policy), index)]
                branch = _routing(record).get("branch") or _branch_outcome(record).get("branch")
                decision = _final_decision(record)
                rows.append(
                    '<tr data-case-row>'
                    f"<td>{_case_link(filename, case_id)}</td>"
                    f"<td>{_esc(_actual(record))}</td>"
                    f"<td>{_esc(_prediction(record) or _proposed_prediction(record))}</td>"
                    f"<td>{_esc(branch)}</td>"
                    f"<td>{_esc(decision.get('action'))}</td>"
                    f'<td><span class="badge {classification}">{_esc(label)}</span></td>'
                    "</tr>"
                )
            body = (
                '<p class="empty">无 case</p>'
                if not rows
                else '<div class="table-wrap"><table><thead><tr>'
                "<th>Case</th><th>真值</th><th>预测/候选</th><th>分支</th><th>M9 动作</th><th>归类</th>"
                "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
            )
            sections.append(
                f'<section class="group"><h3><span class="badge {classification}">{_esc(label)}</span>'
                f" · {_esc(len(items))}</h3>{body}</section>"
            )
    return _document("RCA 离线实验报告", "".join(sections), index=True)


def _feature_tokens(record: Mapping[str, Any]) -> Tuple[List[Any], Mapping[str, Any], str]:
    features = _mapping(record.get("features"))
    tokens = list(_sequence(features.get("tokens") or features.get("signature")))
    by_family = _mapping(features.get("by_family"))
    if not tokens:
        graph = _mapping(record.get("diagnosis_graph"))
        for node in _sequence(graph.get("nodes")):
            item = _mapping(node)
            if str(item.get("type", "")).lower() == "featuretoken":
                attrs = _mapping(item.get("attrs"))
                tokens.append(attrs.get("token") or item.get("id"))
    return _dedupe(tokens), by_family, str(features.get("telemetry_status") or _telemetry_status(record))


def _render_features(record: Mapping[str, Any]) -> str:
    tokens, by_family, telemetry = _feature_tokens(record)
    token_html = (
        '<div class="tokens">'
        + "".join(f'<span class="token">{_esc(token)}</span>' for token in tokens)
        + "</div>"
        if tokens
        else '<p class="empty">无特征 token</p>'
    )
    family_html = _json_block(by_family) if by_family else '<p class="empty">无 family 分组</p>'
    return _kv_table((("遥测状态", telemetry), ("Token 数", len(tokens)))) + token_html + "<h3>按特征族</h3>" + family_html


def _render_candidates(record: Mapping[str, Any]) -> str:
    match = _mapping(record.get("match"))
    candidates = _sequence(match.get("candidates") or match.get("top_candidates"))
    rows = []
    for candidate in candidates:
        item = _mapping(candidate)
        rows.append(
            (
                item.get("case_id"),
                item.get("label"),
                _number(item.get("similarity"), 4),
                _percent(item.get("evidence_coverage")),
                ", ".join(map(str, _sequence(item.get("overlap") or item.get("shared_evidence")))),
                ", ".join(map(str, _sequence(item.get("missing_evidence")))),
                ", ".join(map(str, _sequence(item.get("conflicting_evidence")))),
            )
        )
    overview = _kv_table(
        (
            ("最高相似度", match.get("max_similarity")),
            ("证据覆盖率", match.get("evidence_coverage")),
            ("候选总数", match.get("retrieved_candidate_count", len(candidates))),
            ("标签纯净", match.get("is_label_pure")),
        )
    )
    return overview + _table(("历史 case", "标签", "相似度", "覆盖率", "重叠证据", "缺失证据", "冲突证据"), rows)


def _render_sop(record: Mapping[str, Any]) -> str:
    sop = record.get("sop_prediction")
    if sop is not None:
        return _json_block(sop)
    links = [
        item
        for item in _sequence(_branch_outcome(record).get("evidence_chain"))
        if "sop" in str(_mapping(item).get("kind", "")).lower()
    ]
    if links:
        return _json_block(links)
    return '<p class="empty">该 case 没有 SOP 预测记录</p>'


def _render_evidence(record: Mapping[str, Any]) -> str:
    outcome = _branch_outcome(record)
    links = _sequence(outcome.get("evidence_chain"))
    rows = []
    for link in links:
        item = _mapping(link)
        rows.append(
            (
                item.get("kind"),
                item.get("statement"),
                ", ".join(map(str, _sequence(item.get("tokens")))),
                item.get("source"),
            )
        )
    evidence_pack = record.get("evidence_pack")
    return (
        "<h3>物理约束与证据链</h3>"
        + _table(("类型", "陈述", "证据 token", "来源"), rows)
        + "<h3>Evidence Pack</h3>"
        + _json_block(evidence_pack)
    )


def _missing_evidence(record: Mapping[str, Any]) -> List[Any]:
    values: List[Any] = []
    values.extend(_sequence(_branch_outcome(record).get("missing_evidence")))
    values.extend(_sequence(_final_decision(record).get("requested_evidence")))
    match = _mapping(record.get("match"))
    values.extend(_sequence(match.get("missing_evidence")))
    for candidate in _sequence(match.get("candidates") or match.get("top_candidates")):
        values.extend(_sequence(_mapping(candidate).get("missing_evidence")))
    return _dedupe(values)


def _render_missing(record: Mapping[str, Any]) -> str:
    values = _missing_evidence(record)
    if not values:
        return '<p class="empty">未记录缺失证据</p>'
    return "<ul>" + "".join(f"<li><code>{_esc(value)}</code></li>" for value in values) + "</ul>"


def _violations(check: Any) -> Sequence[Any]:
    mapping = _mapping(check)
    return _sequence(mapping.get("violations") or mapping.get("fatal") or mapping.get("errors"))


def _render_trace(trace: Any) -> str:
    value = _mapping(trace)
    if not value:
        return (
            '<p class="empty">无 trace：该 case 未调用 LLM，或旧实验产物未保存逐轮推理记录。</p>'
        )
    attempts = _sequence(value.get("attempts"))
    chunks = [
        _kv_table(
            (
                ("后端", value.get("backend")),
                ("Prompt 版本", value.get("prompt_version")),
                ("约束库版本", value.get("constraint_library_version")),
                ("轮数", value.get("attempt_count", len(attempts))),
                ("发生重写", value.get("rewrote")),
                ("降级原因", value.get("degradation_reason") or value.get("abstain_reason")),
                ("兜底来源", value.get("fallback_source")),
            )
        )
    ]
    evidence_check = value.get("evidence_check")
    if evidence_check is not None:
        chunks.extend(("<h3>输入证据校验</h3>", _json_block(evidence_check)))
    if not attempts:
        chunks.append('<p class="empty">Trace 存在，但没有生成轮次。</p>')
    for index, attempt in enumerate(attempts, start=1):
        item = _mapping(attempt)
        check = _mapping(item.get("check"))
        violations = _violations(check)
        violation_html = (
            "".join(
                '<div class="violation">'
                f"<strong>{_esc(_mapping(v).get('kind') or 'violation')}</strong> "
                f"[{_esc(_mapping(v).get('severity'))}]：{_esc(_mapping(v).get('message') or v)}"
                f"<div class=\"muted\">{_esc(_mapping(v).get('detail'))}</div></div>"
                for v in violations
            )
            if violations
            else '<p class="ok">本轮未记录校验违规</p>'
        )
        chunks.append(
            '<section class="attempt">'
            f"<h3>LLM 第 {_esc(item.get('index', index - 1))} 轮</h3>"
            f"<p>结构化解析：<strong>{_esc(item.get('parsed'))}</strong></p>"
            "<h3>Prompt</h3>"
            + _json_block(item.get("prompt"))
            + "<h3>Raw 输出</h3>"
            + _json_block(item.get("raw_output"))
            + "<h3>校验违规</h3>"
            + violation_html
            + "</section>"
        )
    chunks.extend(("<h3>接受的结构化输出</h3>", _json_block(value.get("accepted"))))
    return "".join(chunks)


def _render_graph(record: Mapping[str, Any]) -> str:
    graph = _mapping(record.get("diagnosis_graph"))
    if not graph:
        return '<p class="empty">无诊断图</p>'
    nodes = []
    for node in _sequence(graph.get("nodes")):
        item = _mapping(node)
        nodes.append((item.get("id"), item.get("type"), _text(item.get("attrs"))))
    edges = []
    for edge in _sequence(graph.get("edges")):
        item = _mapping(edge)
        edges.append((item.get("src"), item.get("type"), item.get("dst"), _text(item.get("attrs"))))
    return (
        _kv_table(
            (
                ("Case", graph.get("case_id")),
                ("SOP 版本", graph.get("sop_version")),
                ("约束库版本", graph.get("constraint_library_version")),
                ("确认人", graph.get("confirmed_by")),
                ("内容指纹", graph.get("content_hash")),
            )
        )
        + "<h3>节点</h3>"
        + _table(("ID", "类型", "属性"), nodes)
        + "<h3>边</h3>"
        + _table(("源", "关系", "目标", "属性"), edges)
    )


def _case_page(policy: str, record: Mapping[str, Any], trace: Any) -> str:
    case_id = record.get("case_id", "unknown-case")
    decision = _final_decision(record)
    outcome = _branch_outcome(record)
    routing = _routing(record)
    classification = _classification(record)
    match = _mapping(record.get("match"))
    title = f"{policy} / {case_id}"
    overview = _kv_table(
        (
            ("Policy", policy),
            ("Case ID", case_id),
            ("真值", _actual(record)),
            ("最终预测", _prediction(record)),
            ("候选预测", _proposed_prediction(record)),
            (
                "最终是否正确",
                classification == "correct" if _prediction(record) is not None else "未给最终结论",
            ),
            (
                "候选是否正确",
                _proposed_prediction(record) == _actual(record)
                if _proposed_prediction(record) is not None
                else "无候选",
            ),
            ("M9 动作", decision.get("action")),
            ("路由分支", routing.get("branch") or outcome.get("branch")),
            ("路由原因", routing.get("reason")),
            ("相似度 / 覆盖率", f"{_text(match.get('max_similarity'))} / {_text(match.get('evidence_coverage'))}"),
            ("置信度", _percent(decision.get("confidence", outcome.get("confidence")))),
            (
                "Wilson 下界 / 支持数",
                f"{_percent(decision.get('confidence_lower_bound', outcome.get('confidence_lower_bound')))} / "
                f"{_text(decision.get('calibration_support', outcome.get('calibration_support')))}",
            ),
        )
    )
    body = [
        '<nav class="crumb"><a href="../index.html">← 返回实验索引</a></nav>',
        f"<h1>{_esc(title)}</h1>",
        f'<p><span class="badge {classification}">{_esc(_CLASS_LABELS[classification])}</span></p>',
        _panel("结论、真值与路由", overview),
        _panel("特征 Token", _render_features(record)),
        _panel("历史候选", _render_candidates(record)),
        _panel("SOP 预测", _render_sop(record)),
        _panel("物理约束与证据链", _render_evidence(record)),
        _panel("四维置信度与约束扣分", _confidence_breakdown_panel(record)),
        _panel(
            "M9 决策原因",
            _kv_table(
                (
                    ("动作", decision.get("action")),
                    ("原因", decision.get("reason")),
                    ("标定分组", decision.get("calibration_group")),
                    ("支持数", decision.get("calibration_support")),
                    ("最终结论", decision.get("verdict")),
                    ("降级前候选", decision.get("proposed_verdict")),
                )
            ),
        ),
        _panel("缺失证据", _render_missing(record)),
        _panel("LLM 逐轮推理", _render_trace(trace)),
        _panel("诊断图", _render_graph(record)),
        _panel("结构化 RCA 报告", _json_block(record.get("report")), open_panel=False),
    ]
    return _document(title, "".join(body))


def render_experiment_html(
    output_dir: str | Path,
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
    outcomes: Mapping[str, Sequence[Mapping[str, Any]]],
    traces: Mapping[str, Mapping[str, Any]],
    training_summary: Any = None,
) -> Dict[str, Any]:
    """Render an experiment index and one self-contained page per case.

    Args:
        output_dir: Destination directory. ``index.html`` and ``cases/`` are
            created below it.
        summary: Aggregate evaluation summary, either the complete
            ``summary.json`` object or its ``policies`` mapping.
        manifest: Run manifest from ``run_manifest.json``.
        outcomes: Mapping of policy name to evaluate_routing-compatible rows.
        traces: Mapping of policy name to ``case_id -> trace``.
        training_summary: Optional JSON-compatible training diagnostics.

    Returns:
        A JSON-compatible manifest describing all generated HTML files.
    """

    destination = Path(output_dir)
    cases_dir = destination / "cases"
    destination.mkdir(parents=True, exist_ok=True)
    cases_dir.mkdir(parents=True, exist_ok=True)

    summary_map = _mapping(summary)
    manifest_map = _mapping(manifest)
    outcomes_map = _mapping(outcomes)
    traces_map = _mapping(traces)
    case_files: Dict[Tuple[str, int], str] = {}
    generated: List[str] = []
    used_names: set[str] = set()
    policy_counts: Dict[str, int] = {}

    for raw_policy, raw_records in outcomes_map.items():
        policy = str(raw_policy)
        records = [_mapping(item) for item in _sequence(raw_records)]
        policy_counts[policy] = len(records)
        policy_traces = _mapping(traces_map.get(raw_policy))
        for index, record in enumerate(records):
            case_id = str(record.get("case_id") or f"case-{index + 1}")
            filename = _case_filename(policy, case_id, used_names)
            case_files[(policy, index)] = filename
            trace_id = record.get("trace_id")
            trace = policy_traces.get(str(trace_id)) if trace_id not in (None, "") else None
            if trace is None:
                trace = policy_traces.get(case_id)
            path = cases_dir / filename
            path.write_text(_case_page(policy, record, trace), encoding="utf-8")
            generated.append(str(Path("cases") / filename))

    index_path = destination / "index.html"
    index_path.write_text(
        _index_page(
            summary_map,
            manifest_map,
            outcomes_map,
            traces_map,
            case_files,
            training_summary,
        ),
        encoding="utf-8",
    )
    relative_files = ["index.html", *generated]
    return {
        "schema_version": REPORT_SCHEMA,
        "output_dir": str(destination),
        "index": str(index_path),
        "cases_dir": str(cases_dir),
        "case_count": len(generated),
        "file_count": len(relative_files),
        "policies": policy_counts,
        "relative_files": relative_files,
        "case_files": generated,
    }
