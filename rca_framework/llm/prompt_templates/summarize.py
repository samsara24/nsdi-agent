"""High-confidence historical summary prompt.

The high tier already owns the verdict through exact historical reuse.  The LLM
may narrate the evidence, but the schema deliberately has no verdict field.
"""

from __future__ import annotations

import json
from typing import Any


SUMMARY_PROMPT_VERSION = "rca-summary-high-tier-v1"


def build_summary_prompt(request: Any) -> str:
    payload = {
        "case_id": request.case_id,
        "branch": request.branch,
        "routing_reason": request.routing_reason,
        "available_evidence": list(request.evidence_tokens),
        "historical_case_ids": list(request.historical_case_ids),
        "historical_label_distribution": dict(request.historical_label_distribution),
    }
    return "\n\n".join((
        "你是光链路故障定界报告助手。本分支已经由历史完全匹配给出结论，"
        "你的任务只是把当前证据与历史表象整理成可读总结，不能改变结论。",
        json.dumps(payload, ensure_ascii=False, indent=2),
        "只输出 JSON："
        '{"summary":"...","evidence_narrative":[{"token":"...","reading":"..."}],"caveats":["..."]}',
    ))
