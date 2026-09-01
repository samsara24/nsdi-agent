"""M8 LLM 子包。

兼容性：`from rca_framework.llm import PathLLMReasoner` 必须继续可用，
`pipeline.py` 与历史脚本依赖它，它也是 legacy 58/85 回归锚点的一部分。
legacy 实现原样搬进 `legacy.py`，本模块负责这层兼容。

新增的 M8 接口是 `ConstrainedReasoner`：它与 legacy `PathLLMReasoner` 的关键区别是
输出结构可逐步校验（见 `protocol.py`），并且在约束校验失败时会重写或弃权，
而不是把无法校验的自由文本直接当成结论。
"""

from .backend import Backend, NoneBackend, ScriptedBackend, VLLMBackend, backend_for
from .legacy import (
    LLM_OUTPUT_SCHEMA,
    PathLLMReasoner,
    build_path_prompt,
    parse_llm_json,
)
from .prompts import (
    FILTERED_RULE_PROMPT_TEMPLATE_VERSION,
    LEGACY_PROMPT_TEMPLATE_VERSION,
    PROMPT_TEMPLATE_VERSION,
    SOP_VERSION,
    build_prompt,
    prompt_template_hash,
    prompt_template_version_for,
)
from .protocol import (
    DIAGNOSIS_OUTPUT_SCHEMA,
    DiagnosisResponse,
    ReasoningStep,
    parse_response,
)
from .reason import Attempt, ConstrainedReasoner, ReasoningTrace

__all__ = [
    "Attempt",
    "Backend",
    "ConstrainedReasoner",
    "DIAGNOSIS_OUTPUT_SCHEMA",
    "DiagnosisResponse",
    "LLM_OUTPUT_SCHEMA",
    "NoneBackend",
    "FILTERED_RULE_PROMPT_TEMPLATE_VERSION",
    "LEGACY_PROMPT_TEMPLATE_VERSION",
    "PROMPT_TEMPLATE_VERSION",
    "SOP_VERSION",
    "PathLLMReasoner",
    "ReasoningStep",
    "ReasoningTrace",
    "ScriptedBackend",
    "VLLMBackend",
    "backend_for",
    "build_path_prompt",
    "build_prompt",
    "parse_llm_json",
    "parse_response",
    "prompt_template_hash",
    "prompt_template_version_for",
]
