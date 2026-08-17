"""把 107 条测试 case 压缩成一份自包含的报告数据包，供离线 HTML 直接内嵌。

`case_analysis_payload.json` 有 3.6 MB，直接塞进 HTML 会让浏览器解析变慢，而且里面
大量字段（完整 prompt、每次尝试的原始文本、逐候选的证据差集）在报告里用不到。
这里只保留「复现一条 case 的判断链」所需的最小集合：原始 lane 读数、证据 token、
专家逐端诊断、SOP 叶节点、检索邻居、M9 结论，以及 LLM 三个岗位的输出与思维链。

同时补两份 payload 里没有的东西：
  1. 专家规则在**全库 268 条**上的分格可靠性（Wilson 下界 vs 所判类别先验），
     这是判断「某条规则到底该不该给结论」的唯一依据；
  2. 每条 case 落在哪个格子，从而在逐 case 页上能直接标出「这一步本来就不可靠」。
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rca_framework.data import cases_by_manifest_split
from rca_framework.evidence_pack import build_packs, labels_of
from rca_framework.expert import diagnose_many
from rca_framework.types import wilson_lower_bound

DATASET = ROOT / "datasets/rca_v2_l2fixed"
PAYLOAD = ROOT / "artifacts/case_analysis_payload.json"
OUT = ROOT / "artifacts/report_bundle.json"

METRICS = ("rxpower", "txpower", "media_snr", "host_snr", "serdes_snr", "bias")
STATUS_KEYS = ("TxLOS", "TxLOL", "RxLOS", "RxLOL")
SIDES = ("L1", "L2")
PRIOR = {"L2": 0.6231, "L1": 0.3022, "fiber": 0.0746}


def lane_list(case: Dict[str, Any], metric: str, side: str) -> List[Optional[float]]:
    """原始 JSON 里 lane 索引是字符串键；11 条 case 的 bias 整块没被解析成 dict。"""
    block = (case.get(metric) or {}).get(side)
    if isinstance(block, dict):
        return [block[k] for k in sorted(block, key=lambda x: int(x) if str(x).isdigit() else 0)]
    return []


def cell_of(diag: Any) -> str:
    """规则格子 = (胜出规则, 异常端->定界端)。可靠性只能在这个粒度上统计。"""
    if not diag.sides:
        return f"{diag.group}|-"
    best = diag.sides[0]
    return f"{best.rule}|{best.side}->{best.location}"


def reliability_table() -> Dict[str, Any]:
    train = cases_by_manifest_split(DATASET, "train")
    test = cases_by_manifest_split(DATASET, "test")
    every = train + test
    labels = labels_of(every)
    diags = diagnose_many(build_packs(every))

    cells: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "ok": 0, "dist": Counter()}
    )
    for label, diag in zip(labels, diags):
        cell = cells[cell_of(diag)]
        cell["n"] += 1
        cell["ok"] += int(diag.verdict == label)
        cell["dist"][label] += 1

    table = {}
    for key, cell in cells.items():
        verdict_side = key.split("->")[-1] if "->" in key else ""
        prior = PRIOR.get(verdict_side, 0.0)
        lower = wilson_lower_bound(cell["ok"], cell["n"])
        table[key] = {
            "n": cell["n"],
            "ok": cell["ok"],
            "acc": cell["ok"] / cell["n"],
            "wilson_lb": lower,
            "verdict_prior": prior,
            # 「不如直接猜这一类的先验」= 这个格子提供的信息量在统计上无法证实
            "beats_prior": bool(prior and lower > prior),
            "dist": dict(cell["dist"]),
            "fiber_rate": cell["dist"]["fiber"] / cell["n"],
        }
    return table


def main() -> int:
    payload = json.loads(PAYLOAD.read_text(encoding="utf-8"))
    by_id = {c["case_id"]: c for c in payload["cases"]}

    test = cases_by_manifest_split(DATASET, "test")
    packs = build_packs(test)
    labels = labels_of(test)
    diags = diagnose_many(packs)
    table = reliability_table()

    cases: List[Dict[str, Any]] = []
    for case, label, diag in zip(test, labels, diags):
        row = by_id.get(case["case_id"])
        if row is None:
            continue
        final = row["final"]
        cell = cell_of(diag)

        llm = None
        if row.get("llm_diagnosis"):
            diagnosis = row["llm_diagnosis"]
            last = diagnosis["attempts"][-1] if diagnosis["attempts"] else None
            llm = {
                "accepted": diagnosis["accepted"],
                "rewrote": diagnosis["rewrote"],
                "attempt_count": diagnosis["attempt_count"],
                "abstain_reason": diagnosis.get("abstain_reason", ""),
                "attempts": [
                    {
                        "index": a["index"],
                        "parsed": a["parsed"],
                        "thinking": a["thinking"][:6000],
                        "violations": a["violations"],
                        "fatal": a["fatal_count"],
                    }
                    for a in diagnosis["attempts"]
                ],
                "verdict": (last or {}).get("parsed", {}).get("verdict") if last and last.get("parsed") else None,
            }

        cases.append(
            {
                "id": case["case_id"],
                "gold": label,
                "pred": final["verdict"],
                "ok": final["verdict"] == label,
                "source": final.get("source"),
                "final_conf": final.get("confidence"),
                "needs_human": final.get("needs_human"),
                "cell": cell,
                "group": diag.group,
                "reason": diag.reason,
                "expert_verdict": diag.verdict,
                "sides": [
                    {
                        "side": s.side,
                        "rule": s.rule,
                        "loc": s.location,
                        "prio": s.priority,
                        "anom": dict(s.anomalies),
                    }
                    for s in diag.sides
                ],
                "raw": {m: {s: lane_list(case, m, s) for s in SIDES} for m in METRICS},
                "status": {
                    k: {s: (case.get(k) or {}).get(s) for s in SIDES} for k in STATUS_KEYS
                },
                "ctx": {
                    "alarm": case.get("alarm_name"),
                    "lanes": case.get("Lane number"),
                    "alarm_if": bool(case.get("alarm_ip_interface")),
                    "temp": {s: (case.get("Temperature") or {}).get(s) for s in SIDES},
                    "volt": {s: (case.get("Voltage") or {}).get(s) for s in SIDES},
                },
                "tokens": row["features"]["tokens"],
                "sop": (row.get("sop") or {}).get("verdict"),
                "match": {
                    "sim": row["match"]["max_similarity"],
                    "cov": row["match"]["evidence_coverage"],
                    "ties": row["match"]["tie_count"],
                    "cands": [
                        {"id": c["case_id"], "label": c["label"], "sim": c["similarity"]}
                        for c in row["match"]["candidates"][:5]
                    ],
                },
                "routing": row["routing"].get("branch") if isinstance(row.get("routing"), dict) else None,
                "branch": {
                    k: row["branch_outcome"].get(k)
                    for k in ("branch", "verdict", "confidence", "confidence_lower_bound",
                              "calibration_group", "calibration_support", "needs_llm", "needs_human")
                },
                "llm": llm,
                "challenge": row.get("llm_challenge"),
                "explain": row.get("llm_explain"),
            }
        )

    bundle = {
        "meta": payload["meta"],
        "metrics": payload["metrics"],
        "challenge_summary": payload["challenge_summary"],
        "explain_summary": payload["explain_summary"],
        "reliability": table,
        "cases": cases,
    }
    OUT.write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    size = OUT.stat().st_size / 1e6
    ok = sum(1 for c in cases if c["ok"])
    answered = sum(1 for c in cases if c["pred"] is not None)
    print(f"wrote {OUT} ({size:.2f} MB) cases={len(cases)} answered={answered} correct={ok}")
    print(f"reliability cells={len(table)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
