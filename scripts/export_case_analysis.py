"""把 107 条测试 case 的完整决策链导出为一份自包含 JSON，供离线 HTML 报告使用。

导出的是「一条 case 从原始遥测走到最终结论」的全链路快照：原始 lane 读数、
抽取出的证据 token、专家规则的逐端诊断与两端裁决、SOP 叶节点、证据图检索命中、
M9 候选级联，以及 LLM 在三个岗位（定界 / 质疑 / 解释）上对同一条 case 的输出。

之所以要合成一份文件：这四份产物分别来自四次实验（i3b / i4v2 / i5），
按 case_id 对齐之后才能回答「同一条 case 上规则怎么判、模型怎么想、谁错了」。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rca_framework.llm.protocol import parse_response

I3B = ROOT / "artifacts/i3b_offline_expert_v3_risk035"
I4V2 = ROOT / "artifacts/i4_challenge_v2"
I5 = ROOT / "artifacts/i5_explain_v1"

METRICS = ("rxpower", "txpower", "media_snr", "host_snr", "serdes_snr", "bias")
STATUS_KEYS = ("TxLOS", "TxLOL", "RxLOS", "RxLOL")
SIDES = ("L1", "L2")


def _lane_map(block: Any, side: str) -> Dict[str, Optional[float]]:
    if not isinstance(block, dict):
        return {}
    lanes = block.get(side)
    if isinstance(lanes, dict):
        return {str(k): lanes[k] for k in sorted(lanes, key=lambda x: str(x))}
    if isinstance(lanes, list):
        return {str(i): v for i, v in enumerate(lanes)}
    return {}


def telemetry_view(pack: Dict[str, Any]) -> Dict[str, Any]:
    tel = pack["telemetry"]
    metrics: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for metric in METRICS:
        metrics[metric] = {side: _lane_map(tel.get(metric), side) for side in SIDES}
    return {
        "metrics": metrics,
        "statuses": {key: {side: (tel.get(key) or {}).get(side) for side in SIDES} for key in STATUS_KEYS},
        "scalars": {
            key: {side: (tel.get(key) or {}).get(side) for side in SIDES}
            for key in ("Temperature", "Voltage")
        },
        "context": {
            "alarm_name": tel.get("alarm_name"),
            "alarm_time": tel.get("alarm_time"),
            "alarm_ip_interface": tel.get("alarm_ip_interface"),
            "link_side_ip_interface_map": tel.get("link_side_ip_interface_map"),
            "lane_number": tel.get("Lane number"),
            "vendor": tel.get("vendor"),
            "region": tel.get("region"),
        },
        "telemetry_status": pack.get("telemetry_status"),
        "coverage": pack.get("coverage"),
        "missing_fields": pack.get("missing_fields", []),
        "optical_blackout": pack.get("optical_blackout"),
    }


def llm_steps(raw_output: str) -> Optional[Dict[str, Any]]:
    """从一次生成里取出结构化推理链。校验器只存了违规，没存解析结果。"""
    response = parse_response(raw_output)
    if response is None:
        return None
    return {
        "verdict": response.verdict,
        "confidence": response.confidence,
        "missing_information": list(response.missing_information),
        "steps": [
            {
                "claim": step.claim,
                "cited_evidence": list(step.cited_evidence),
                "cited_constraints": list(step.cited_constraints),
                "effect": step.effect,
                "target": step.target,
            }
            for step in response.steps
        ],
    }


def _thinking(raw_output: str) -> str:
    """蒸馏模型把思维链放在 <think> 块里，它是「模型如何得到结论」的唯一记录。"""
    start = raw_output.find("<think>")
    end = raw_output.find("</think>")
    if start < 0 or end < 0:
        return ""
    return raw_output[start + len("<think>"): end].strip()


def main() -> int:
    outcomes = json.loads((I3B / "outcomes.json").read_text(encoding="utf-8"))["coverage-v2"]
    traces = json.loads((I3B / "traces.json").read_text(encoding="utf-8"))["coverage-v2"]
    summary = json.loads((I3B / "summary.json").read_text(encoding="utf-8"))
    challenge = {r["case_id"]: r for r in json.loads((I4V2 / "records.json").read_text(encoding="utf-8"))}
    explain = {r["case_id"]: r for r in json.loads((I5 / "records.json").read_text(encoding="utf-8"))}

    cases: List[Dict[str, Any]] = []
    for outcome in outcomes:
        case_id = outcome["case_id"]
        pack = outcome["evidence_pack"]
        trace = traces.get(case_id)

        llm_diag = None
        if trace is not None:
            attempts = []
            for attempt in trace["attempts"]:
                attempts.append({
                    "index": attempt["index"],
                    "parsed": llm_steps(attempt["raw_output"]),
                    "thinking": _thinking(attempt["raw_output"]),
                    "violations": [
                        {
                            "kind": v["kind"],
                            "severity": v["severity"],
                            "message": v["message"],
                            "constraint_id": v.get("constraint_id", ""),
                            "step_index": v.get("step_index"),
                            "detail": v.get("detail", ""),
                        }
                        for v in attempt["check"]["violations"]
                    ],
                    "fatal_count": attempt["check"]["fatal_count"],
                    "ok": attempt["check"]["ok"],
                })
            llm_diag = {
                "prompt_version": trace["prompt_version"],
                "constraint_library_version": trace["constraint_library_version"],
                "attempt_count": trace["attempt_count"],
                "rewrote": trace["rewrote"],
                "accepted": trace["accepted"] is not None,
                "abstain_reason": trace.get("abstain_reason") or "",
                "attempts": attempts,
                # prompt 只存一份：两次尝试只差 retry_feedback 段
                "prompt": trace["attempts"][0]["prompt"],
            }

        challenge_row = challenge.get(case_id)
        explain_row = explain.get(case_id)

        cases.append({
            "case_id": case_id,
            "gold": outcome["actual"],
            "telemetry": telemetry_view(pack),
            "features": {
                "tokens": outcome["features"]["tokens"],
                "by_family": outcome["features"]["by_family"],
            },
            "expert": outcome["expert_diagnosis"],
            "expert_prediction": outcome["expert_prediction"],
            "sop": outcome["sop_prediction"],
            "match": {
                "max_similarity": outcome["match"]["max_similarity"],
                "evidence_coverage": outcome["match"]["evidence_coverage"],
                "tie_count": outcome["match"]["tie_count"],
                "candidates": [
                    {
                        "case_id": c["case_id"],
                        "label": c["label"],
                        "similarity": c["similarity"],
                        "shared_evidence": c.get("shared_evidence", []),
                        "missing_evidence": c.get("missing_evidence", []),
                        "extra_evidence": c.get("extra_evidence", []),
                    }
                    for c in outcome["match"]["candidates"][:5]
                ],
            },
            "routing": outcome["routing"],
            "branch_outcome": {
                key: outcome["branch_outcome"][key]
                for key in (
                    "branch", "verdict", "confidence", "confidence_lower_bound",
                    "calibration_group", "calibration_support", "needs_llm",
                    "needs_human", "caveats", "missing_evidence",
                )
                if key in outcome["branch_outcome"]
            },
            "final": outcome["final_decision"],
            "llm_diagnosis": llm_diag,
            "llm_challenge": None if challenge_row is None else {
                "score": challenge_row["score"],
                "challenged": challenge_row["challenged"],
                "response": challenge_row["response"],
                "thinking": _thinking(challenge_row["raw_output"]),
            },
            "llm_explain": None if explain_row is None else {
                "explanation": explain_row["explanation"],
                "checks": explain_row["checks"],
                "thinking": _thinking(explain_row["raw_output"]),
            },
        })

    payload = {
        "meta": {
            "run": "i3b_offline_expert_v3_risk035 + i4_challenge_v2 + i5_explain_v1",
            "model": "DeepSeek-R1-Distill-Qwen-32B",
            "dataset": "rca_v2_l2fixed",
            "split": "manifest train/test = 161/107",
            "prompt_version": "rca-constrained-reasoning-v7",
            "constraint_library_version": "constraint-library-v6",
            "feature_profile": "v3",
        },
        "metrics": summary["policies"]["coverage-v2"]["final_decisions"],
        "challenge_summary": json.loads((I4V2 / "summary.json").read_text(encoding="utf-8")),
        "explain_summary": json.loads((I5 / "summary.json").read_text(encoding="utf-8")),
        "cases": cases,
    }

    out = ROOT / "artifacts/case_analysis_payload.json"
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB), {len(cases)} cases")

    right = sum(1 for c in cases if c["final"]["verdict"] == c["gold"])
    answered = sum(1 for c in cases if c["final"]["verdict"] is not None)
    with_llm = sum(1 for c in cases if c["llm_diagnosis"] is not None)
    print(f"answered={answered} correct={right} llm_traces={with_llm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
