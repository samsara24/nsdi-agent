"""把已保存的 LLM 回答重放过校验器，量化校验器改动的效果。

用途：校验器的表达能力（哪些引用方式算合规）与模型无关，因此不需要重跑 GPU
就能评估改动。输入是某次实验的 `traces.json`，输出是新旧两套判定下的
通过率、违规构成，以及**通过与判对是否相关**——后者才是校验器存在的理由。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca_framework.constraints.checker import check_response
from rca_framework.constraints.library import CONSTRAINT_LIBRARY
from rca_framework.data import cases_by_manifest_split
from rca_framework.evidence_pack import build_packs, labels_of
from rca_framework.llm.protocol import parse_response


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _available_evidence(attempt: Dict[str, Any]) -> List[str]:
    """从 prompt 的 case 描述块里回捞证据清单（它是一段 JSON 数组）。"""
    prompt = attempt.get("prompt", "")
    block = re.search(r'"可用证据":\s*\[(.*?)\]', prompt, flags=re.S)
    if block:
        return re.findall(r'"([^"]+)"', block.group(1))
    for violation in attempt["check"]["violations"]:
        detail = violation.get("detail", "")
        if detail.startswith("可用证据="):
            return re.findall(r"'([^']+)'", detail)
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/rca_v2_l2fixed"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--policy", default="coverage-v2")
    args = parser.parse_args()

    traces = json.loads(args.traces.read_text(encoding="utf-8"))[args.policy]
    cases = cases_by_manifest_split(args.data_dir, args.split)
    packs = build_packs(cases)
    gold = dict(zip((pack.case_id for pack in packs), labels_of(cases)))
    by_id = {pack.case_id: pack for pack in packs}

    old_ok = new_ok = 0
    old_kinds: Counter = Counter()
    new_kinds: Counter = Counter()
    rows = []
    for case_id, trace in traces.items():
        pack = by_id.get(case_id)
        if pack is None:
            continue
        accepted_old = bool(trace["accepted"])
        accepted_new = False
        verdict = None
        for attempt in trace["attempts"]:
            response = parse_response(attempt["raw_output"])
            if response is None:
                continue
            report = check_response(
                response, pack, _available_evidence(attempt), library=CONSTRAINT_LIBRARY
            )
            for violation in attempt["check"]["violations"]:
                if violation["severity"] == "fatal":
                    old_kinds[violation["message"][:36]] += 1
            for violation in report.fatal:
                new_kinds[violation.message[:36]] += 1
            if report.ok:
                accepted_new = True
                verdict = response.verdict
                break
            verdict = response.verdict
        old_ok += accepted_old
        new_ok += accepted_new
        rows.append((case_id, accepted_old, accepted_new, verdict, gold.get(case_id)))

    total = len(rows)
    print(f"重放 {total} 条（{args.split}）")
    print(f"  旧校验器通过 {old_ok} = {old_ok / total:.1%}")
    print(f"  新校验器通过 {new_ok} = {new_ok / total:.1%}")

    def accuracy(subset: List[Any]) -> str:
        answered = [row for row in subset if row[3] not in (None, "abstain")]
        if not answered:
            return "n=0"
        correct = sum(1 for row in answered if row[3] == row[4])
        return f"{correct}/{len(answered)} = {correct / len(answered):.1%}"

    print(f"  新通过组的判对率: {accuracy([r for r in rows if r[2]])}")
    print(f"  新判废组的判对率: {accuracy([r for r in rows if not r[2]])}")
    print("\n旧 fatal 违规构成:")
    for message, count in old_kinds.most_common(6):
        print(f"  {count:5d}  {message}")
    print("\n新 fatal 违规构成:")
    for message, count in new_kinds.most_common(6):
        print(f"  {count:5d}  {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
