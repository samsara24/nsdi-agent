#!/usr/bin/env python3
"""Archive a formal offline run into experiments/<date>_<name>/ Loop artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from rca_framework.branches.partial import critical_missing, physical_key_reasons
from rca_framework.constraints.library import CONSTRAINT_LIBRARY
from rca_framework.constraints.measurement import MEASUREMENT_CONTRACT_LIBRARY
from rca_framework.constraints.physics import PHYSICS_LIBRARY
from rca_framework.llm.prompts import PROMPT_TEMPLATE_VERSION, prompt_template_hash
from rca_framework.sop import EXPERT_SOP_VERSION, expert_sop_hash
from rca_framework.sop.library import LEARNED_SOP_VERSION


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _policy_records(outcomes: Mapping[str, Any]) -> tuple[str, List[Dict[str, Any]]]:
    if not outcomes:
        raise ValueError("outcomes.json is empty")
    policy = next(iter(outcomes))
    records = outcomes[policy]
    if not isinstance(records, list):
        raise ValueError(f"outcomes[{policy}] is not a list")
    return policy, records


def _failed_step(record: Mapping[str, Any]) -> str:
    routing = record.get("routing") or {}
    branch = (record.get("branch_outcome") or {}).get("branch") or routing.get("branch")
    final = record.get("final_decision") or {}
    action = final.get("action")
    bo = record.get("branch_outcome") or {}
    missing = tuple(bo.get("missing_evidence") or ())
    critical = critical_missing(missing)
    if branch == "N5a":
        if not bo.get("is_label_pure", True) and (bo.get("calibration_group") or "").endswith("mixed"):
            return "N5a_mixed_arbitration"
        return "N5a_historical_reuse"
    if branch == "N5b":
        return "N5b_critical_key_evidence" if critical else "N5b_minor_reuse"
    if branch == "N5c":
        return "N5c_expert_sop_llm"
    if action == "request_evidence":
        return "N6_request_evidence"
    if action == "human_review":
        return "N6_human_review"
    return branch or "unknown"


def _error_class(record: Mapping[str, Any], *, pred: Optional[str], actual: str) -> str:
    branch = ((record.get("branch_outcome") or {}).get("branch") or "")
    tokens = list(((record.get("features") or {}).get("tokens") or []))
    if actual == "fiber" or pred == "fiber":
        return "fiber_not_identifiable_or_hallucinated"
    if branch == "N5a":
        return "exact_match_same_evidence_different_label"
    if branch == "N5b":
        missing = tuple((record.get("branch_outcome") or {}).get("missing_evidence") or ())
        if critical_missing(missing):
            return "critical_missing_arbitration_failed"
        return "noncritical_reuse_wrong_history"
    if branch == "N5c":
        return "cold_start_llm_or_gate_failed"
    if not tokens:
        return "empty_evidence"
    return "other"


def _analyze_cases(
    records: Sequence[Mapping[str, Any]],
    *,
    previous_label_suspects: Sequence[Mapping[str, Any]],
    previous_irreducible: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    prev_suspect_ids = {item.get("case_id") for item in previous_label_suspects}
    prev_irreducible_ids = {item.get("case_id") for item in previous_irreducible}

    case_analysis: List[Dict[str, Any]] = []
    bad_cases: List[Dict[str, Any]] = []
    label_suspects: List[Dict[str, Any]] = list(previous_label_suspects)
    irreducible: List[Dict[str, Any]] = list(previous_irreducible)
    correct_by_branch: Dict[str, Counter] = defaultdict(Counter)
    correct_step_counter: Counter = Counter()
    bad_step_counter: Counter = Counter()

    seen_suspect = set(prev_suspect_ids)
    seen_irreducible = set(prev_irreducible_ids)

    for record in records:
        actual = str(record.get("actual"))
        bo = record.get("branch_outcome") or {}
        final = record.get("final_decision") or {}
        routing = record.get("routing") or {}
        match = record.get("match") or {}
        branch = bo.get("branch") or routing.get("branch")
        action = final.get("action")
        pred = final.get("verdict")
        branch_pred = bo.get("verdict")
        missing = tuple(bo.get("missing_evidence") or ())
        critical = critical_missing(missing)
        failed_step = _failed_step(record)
        tokens = list(((record.get("features") or {}).get("tokens") or []))

        is_final_correct = action == "final" and pred == actual
        is_branch_correct = branch_pred is not None and branch_pred == actual
        status = (
            "correct_final"
            if is_final_correct
            else "bad_final"
            if action == "final" and pred != actual
            else "deferred"
        )

        analysis = {
            "case_id": record.get("case_id"),
            "actual": actual,
            "branch": branch,
            "routing_reason": routing.get("reason"),
            "max_similarity": match.get("max_similarity"),
            "evidence_coverage": match.get("evidence_coverage"),
            "branch_verdict": branch_pred,
            "final_action": action,
            "final_verdict": pred,
            "calibration_group": bo.get("calibration_group") or final.get("calibration_group"),
            "missing_evidence": list(missing),
            "critical_missing": list(critical),
            "critical_reasons": {
                token: list(physical_key_reasons(token)) for token in critical
            },
            "status": status,
            "branch_correct": is_branch_correct,
            "failed_or_success_step": failed_step,
            "tokens": tokens,
            "caveats": list(bo.get("caveats") or []),
            "final_reason": final.get("reason"),
        }
        case_analysis.append(analysis)

        if is_final_correct or (action != "final" and is_branch_correct):
            correct_by_branch[str(branch)]["n"] += 1
            if is_branch_correct:
                correct_by_branch[str(branch)]["branch_correct"] += 1
            if is_final_correct:
                correct_by_branch[str(branch)]["final_correct"] += 1
            correct_step_counter[failed_step] += 1

        branch_wrong = branch_pred is not None and branch_pred != actual
        final_wrong = action == "final" and pred is not None and pred != actual
        if branch_wrong or final_wrong:
            err = _error_class(record, pred=pred or branch_pred, actual=actual)
            fixable = "maybe"
            next_action = "keep_observing"
            if err == "exact_match_same_evidence_different_label":
                fixable = "no_safe_model_fix"
                next_action = "mark_irreducible_or_label_review"
            elif err == "fiber_not_identifiable_or_hallucinated":
                fixable = "no_with_current_telemetry"
                next_action = "keep_as_irreducible_request_otdr"
            elif err == "critical_missing_arbitration_failed":
                fixable = "prompt_or_llm_arbitration"
                next_action = "inspect_llm_trace_and_prompt"
            elif err == "noncritical_reuse_wrong_history":
                fixable = "threshold_or_key_evidence"
                next_action = "revisit_key_evidence_contract"

            bad = {
                "case_id": record.get("case_id"),
                "actual": actual,
                "branch_pred": branch_pred,
                "final_pred": pred,
                "final_action": action,
                "branch": branch,
                "failed_step": failed_step,
                "error_class": err,
                "missing_evidence": list(missing),
                "critical_missing": list(critical),
                "fixable": fixable,
                "next_action": next_action,
                "notes": list(bo.get("caveats") or [])[:3],
            }
            bad_cases.append(bad)
            bad_step_counter[failed_step] += 1

            # Same evidence, different label: label suspect + irreducible.
            if err == "exact_match_same_evidence_different_label":
                cid = record.get("case_id")
                if cid not in seen_suspect:
                    label_suspects.append(
                        {
                            "case_id": cid,
                            "actual": actual,
                            "system_pred": branch_pred or pred,
                            "reason": "N5a exact signature matches a historical case with a different label",
                            "tokens": tokens,
                            "source_experiment": "current",
                        }
                    )
                    seen_suspect.add(cid)
                if cid not in seen_irreducible:
                    irreducible.append(
                        {
                            "case_id": cid,
                            "actual": actual,
                            "reason": "Identical train/test evidence signature with conflicting root cause",
                            "failed_step": failed_step,
                            "source_experiment": "current",
                        }
                    )
                    seen_irreducible.add(cid)
            if err == "fiber_not_identifiable_or_hallucinated" and actual == "fiber":
                cid = record.get("case_id")
                if cid not in seen_irreducible:
                    irreducible.append(
                        {
                            "case_id": cid,
                            "actual": actual,
                            "reason": "C20: fiber not identifiable from current two-end telemetry",
                            "failed_step": failed_step,
                            "source_experiment": "current",
                        }
                    )
                    seen_irreducible.add(cid)

    return {
        "case_analysis": case_analysis,
        "bad_cases": bad_cases,
        "label_suspects": label_suspects,
        "irreducible_cases": irreducible,
        "correct_by_branch": {k: dict(v) for k, v in correct_by_branch.items()},
        "correct_step_counter": dict(correct_step_counter),
        "bad_step_counter": dict(bad_step_counter),
    }


def _render_report_html(
    *,
    experiment_name: str,
    hypothesis: str,
    change_classes: Sequence[str],
    change_notes: Sequence[str],
    summary: Mapping[str, Any],
    analysis: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> str:
    flow = """
N1 EvidencePack → N2 Features → N3 EvidenceGraph Match → N4 Route
  ├─ N5a exact: reuse historical evidence chain
  ├─ N5b partial: physics key-evidence → LLM arbitration  ⟵ 本轮调整
  └─ N5c cold: expert SOP constrains LLM                 ⟵ 本轮调整(prompt)
→ N6 confidence gate → N7 report   (N8 frozen)
""".strip()
    bad_rows = []
    for item in analysis["bad_cases"]:
        bad_rows.append(
            "<tr>"
            f"<td>{item['case_id']}</td>"
            f"<td>{item['actual']}</td>"
            f"<td>{item.get('branch_pred')}</td>"
            f"<td>{item.get('final_pred')}</td>"
            f"<td>{item['branch']}</td>"
            f"<td>{item['failed_step']}</td>"
            f"<td>{item['error_class']}</td>"
            f"<td>{item['fixable']}</td>"
            f"<td>{item['next_action']}</td>"
            "</tr>"
        )
    suspect_rows = "".join(
        f"<li><code>{s.get('case_id')}</code>: {s.get('reason')}</li>"
        for s in analysis["label_suspects"]
    ) or "<li>（无）</li>"
    irreducible_rows = "".join(
        f"<li><code>{s.get('case_id')}</code>: {s.get('reason')}</li>"
        for s in analysis["irreducible_cases"]
    ) or "<li>（无）</li>"
    change_li = "".join(f"<li>{c}</li>" for c in change_notes)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <title>{experiment_name}</title>
  <style>
    body {{ font-family: "IBM Plex Sans", "Noto Sans SC", sans-serif; margin: 24px; color: #1a1a1a; background: linear-gradient(#f7f3ea, #eef2f4); }}
    h1,h2 {{ font-family: "Source Serif 4", "Noto Serif SC", serif; }}
    pre,code {{ background: #fff; border: 1px solid #d7d2c8; padding: 2px 6px; }}
    pre {{ padding: 12px; overflow: auto; }}
    table {{ border-collapse: collapse; width: 100%; background: #fff; }}
    th,td {{ border: 1px solid #d7d2c8; padding: 6px 8px; font-size: 13px; vertical-align: top; }}
    th {{ background: #f0ebe0; text-align: left; }}
    .tag {{ display: inline-block; background: #244a3a; color: #fff; padding: 2px 8px; margin-right: 6px; }}
    .warn {{ background: #7a3e1d; }}
  </style>
</head>
<body>
  <h1>{experiment_name}</h1>
  <p>
    <span class="tag">N8 frozen</span>
    <span class="tag">candidate_order=branch</span>
    {" ".join(f'<span class="tag">{c}</span>' for c in change_classes)}
  </p>
  <h2>1. 当前主流程图与本轮调整点</h2>
  <pre>{flow}</pre>
  <ul>{change_li}</ul>
  <p><b>假设：</b>{hypothesis}</p>

  <h2>2. 当前证据图状态</h2>
  <pre>{json.dumps(manifest.get("evidence_graph", {}), ensure_ascii=False, indent=2)}</pre>

  <h2>3. 当前物理约束与量测契约</h2>
  <pre>{json.dumps(manifest.get("constraints", {}), ensure_ascii=False, indent=2)}</pre>

  <h2>4. SOP 是否变化</h2>
  <pre>{json.dumps(manifest.get("sop", {}), ensure_ascii=False, indent=2)}</pre>

  <h2>5. prompt / threshold / candidate_order / N8</h2>
  <pre>{json.dumps(manifest.get("decision_and_prompt", {}), ensure_ascii=False, indent=2)}</pre>

  <h2>6. 正确 case 分支统计与做对步骤</h2>
  <pre>{json.dumps({
    "correct_by_branch": analysis.get("correct_by_branch"),
    "correct_step_counter": analysis.get("correct_step_counter"),
    "summary_metrics": summary,
  }, ensure_ascii=False, indent=2)}</pre>

  <h2>7. Bad Case 逐条分析</h2>
  <table>
    <thead>
      <tr><th>case_id</th><th>actual</th><th>branch_pred</th><th>final_pred</th><th>branch</th><th>failed_step</th><th>error_class</th><th>fixable</th><th>next</th></tr>
    </thead>
    <tbody>
      {''.join(bad_rows) or '<tr><td colspan="9">无 final/branch bad case</td></tr>'}
    </tbody>
  </table>
  <pre>bad_step_counter = {json.dumps(analysis.get("bad_step_counter"), ensure_ascii=False)}</pre>

  <h2>8. 疑似 label 与 irreducible</h2>
  <h3 class="warn">label_suspects</h3>
  <ul>{suspect_rows}</ul>
  <h3 class="warn">irreducible_cases</h3>
  <ul>{irreducible_rows}</ul>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--short-name", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument(
        "--change-class",
        action="append",
        default=[],
        choices=("evidence_graph", "threshold_or_routing", "llm_prompt", "bug_fix"),
    )
    parser.add_argument("--change-note", action="append", default=[])
    parser.add_argument("--previous-experiment-dir", type=Path, default=None)
    parser.add_argument("--policy", default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    exp_dir = args.experiment_dir.resolve()
    exp_dir.mkdir(parents=True, exist_ok=True)

    outcomes = _read_json(run_dir / "outcomes.json")
    summary_raw = _read_json(run_dir / "summary.json")
    run_manifest = _read_json(run_dir / "run_manifest.json")
    policy, records = _policy_records(outcomes)
    if args.policy:
        policy = args.policy
        records = outcomes[policy]

    prev_suspects: List[Dict[str, Any]] = []
    prev_irreducible: List[Dict[str, Any]] = []
    if args.previous_experiment_dir and args.previous_experiment_dir.exists():
        prev_suspects = _read_json(args.previous_experiment_dir / "label_suspects.json").get(
            "label_suspects", []
        )
        prev_irreducible = _read_json(args.previous_experiment_dir / "irreducible_cases.json").get(
            "irreducible_cases", []
        )

    analysis = _analyze_cases(
        records,
        previous_label_suspects=prev_suspects,
        previous_irreducible=prev_irreducible,
    )

    policy_summary = (summary_raw.get("policies") or {}).get(policy) or summary_raw
    summary = {
        "schema_version": "rca-loop-summary-v1",
        "experiment": args.short_name,
        "source_run_dir": str(run_dir),
        "policy": policy,
        "created_at_utc": _utc_now(),
        "routing": policy_summary.get("routing") or policy_summary.get("routing_distribution"),
        "decision": policy_summary.get("decision"),
        "answered": policy_summary.get("answered"),
        "correct": policy_summary.get("correct"),
        "degeneracy_guard": policy_summary.get("degeneracy_guard"),
        "personal_alignment_gate": policy_summary.get("personal_alignment_gate"),
        "correct_by_branch": analysis["correct_by_branch"],
        "correct_step_counter": analysis["correct_step_counter"],
        "bad_case_count": len(analysis["bad_cases"]),
        "bad_step_counter": analysis["bad_step_counter"],
        "label_suspect_count": len(analysis["label_suspects"]),
        "irreducible_count": len(analysis["irreducible_cases"]),
        "raw_policy_summary_keys": sorted(policy_summary.keys()),
    }

    knowledge = run_manifest.get("knowledge") or {}
    scope = run_manifest.get("scope") or {}
    decision = run_manifest.get("decision") or {}
    llm = run_manifest.get("llm") or {}

    experiment_manifest = {
        "schema_version": "rca-loop-experiment-manifest-v1",
        "experiment_id": f"{exp_dir.name}",
        "created_at_utc": _utc_now(),
        "hypothesis": args.hypothesis,
        "change_classes": args.change_class or ["bug_fix"],
        "change_notes": args.change_note,
        "n8_feedback_update": False,
        "source_run_dir": str(run_dir),
        "source_run_manifest": run_manifest,
        "evidence_graph": {
            "version": knowledge.get("evidence_graph_version"),
            "diagnosis_count": knowledge.get("evidence_graph_diagnosis_count"),
            "feature_dictionary_version": knowledge.get("feature_dictionary_version"),
            "feature_dictionary_hash": knowledge.get("feature_dictionary_hash"),
            "historical_vector_count": knowledge.get("historical_vector_count"),
        },
        "constraints": {
            "library_version": knowledge.get("constraint_library_version") or CONSTRAINT_LIBRARY.version,
            "library_hash": knowledge.get("constraint_library_hash") or CONSTRAINT_LIBRARY.content_hash(),
            "physics_version": PHYSICS_LIBRARY.version,
            "physics_hash": PHYSICS_LIBRARY.content_hash(),
            "measurement_version": MEASUREMENT_CONTRACT_LIBRARY.version,
            "measurement_hash": MEASUREMENT_CONTRACT_LIBRARY.content_hash(),
            "note": "N5b key-evidence uses attribution physics only; measurement contracts are vetoes",
        },
        "sop": {
            "expert_sop_version": knowledge.get("expert_sop_version") or EXPERT_SOP_VERSION,
            "expert_sop_hash": knowledge.get("expert_sop_hash") or expert_sop_hash(),
            "learned_sop_version": knowledge.get("learned_sop_version") or LEARNED_SOP_VERSION,
            "learned_sop_hash": knowledge.get("learned_sop_hash"),
            "expert_sop_changed": False,
            "learned_sop_changed": False,
        },
        "decision_and_prompt": {
            "prompt_template": llm.get("prompt_template") or PROMPT_TEMPLATE_VERSION,
            "prompt_template_hash": llm.get("prompt_template_hash") or prompt_template_hash(),
            "candidate_order": decision.get("candidate_order") or ["branch"],
            "feedback_update": scope.get("feedback_update", False),
            "n8_frozen": True,
        },
    }

    _write_json(exp_dir / "experiment_manifest.json", experiment_manifest)
    _write_json(exp_dir / "summary.json", summary)
    _write_json(
        exp_dir / "case_analysis.json",
        {"schema_version": "rca-loop-case-analysis-v1", "cases": analysis["case_analysis"]},
    )
    _write_json(
        exp_dir / "bad_cases.json",
        {"schema_version": "rca-loop-bad-cases-v1", "bad_cases": analysis["bad_cases"]},
    )
    _write_json(
        exp_dir / "label_suspects.json",
        {"schema_version": "rca-loop-label-suspects-v1", "label_suspects": analysis["label_suspects"]},
    )
    _write_json(
        exp_dir / "irreducible_cases.json",
        {
            "schema_version": "rca-loop-irreducible-cases-v1",
            "irreducible_cases": analysis["irreducible_cases"],
        },
    )
    (exp_dir / "report.html").write_text(
        _render_report_html(
            experiment_name=exp_dir.name,
            hypothesis=args.hypothesis,
            change_classes=args.change_class or ["bug_fix"],
            change_notes=args.change_note,
            summary=summary,
            analysis=analysis,
            manifest=experiment_manifest,
        ),
        encoding="utf-8",
    )
    print(f"archived -> {exp_dir}")


if __name__ == "__main__":
    main()
