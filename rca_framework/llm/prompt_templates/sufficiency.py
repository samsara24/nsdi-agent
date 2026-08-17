"""Mid-tier evidence sufficiency prompt."""

from __future__ import annotations

import json
from typing import Any

from ...constraints.measurement import render_measurement_prompt_block
from ...constraints.physics import render_physics_prompt_block


SUFFICIENCY_PROMPT_VERSION = "rca-sufficiency-mid-tier-v1"


def build_sufficiency_prompt(request: Any) -> str:
    payload = {
        "case_id": request.case_id,
        "branch": request.branch,
        "routing_reason": request.routing_reason,
        "available_evidence": list(request.evidence_tokens),
        "missing_evidence": list(request.missing_fields),
        "historical_case_ids": list(request.historical_case_ids),
        "historical_label_distribution": dict(request.historical_label_distribution),
    }
    return "\n\n".join((
        "你是光链路故障定界的证据充分性评估器。本分支与历史 case 有覆盖关系，"
        "但可能缺关键证据。你只判断缺失证据是否关键、需要补采什么，不输出根因 verdict。",
        render_physics_prompt_block(),
        render_measurement_prompt_block(),
        json.dumps(payload, ensure_ascii=False, indent=2),
        "只输出 JSON："
        '{"assessments":[{"token":"...","is_critical":true,'
        '"physics_reason":"...","cited_constraints":["..."]}],'
        '"evidence_request":["..."]}',
    ))
