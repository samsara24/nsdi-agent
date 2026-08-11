"""M8 固定 prompt 模板。

三条原则：

1. **prompt 不逐 case 手写。** 全部由 `DiagnosisRequest` 与约束库渲染，
   模板本身有版本号并进 `run_manifest.json`。手写 prompt 会让实验不可复现。
2. **约束按类型排序注入：exclusion -> caveat -> invariant -> indicator。**
   先给能排除的，再给不许推的，最后才给提高可能性的。理由见 T2：
   这套数据里「倾向性证据」很容易把模型推向多数类，先做排除可以在加权之前砍掉不可能的选项。
   这个顺序由 `constraints.library.render_prompt_block` 保证，有测试锁定。
3. **弃权必须是被明确允许的选项。** 如果 prompt 只给三个根因，模型一定会三选一。
   阶段 1 已经证明「零证据也给个答案」是 legacy 的主要失败模式，
   所以 schema 里有 `abstain`，prompt 里也必须写清楚什么时候该用它。
"""

from __future__ import annotations

import json
import hashlib
import inspect
from typing import Any, Dict, Optional, Sequence

from ..constraints.library import CONSTRAINT_LIBRARY, ConstraintLibrary, render_prompt_block
from ..types import ROOT_CAUSES


#: v2 改写了弃权判据。v1 只说「证据不足就弃权」，真机实测（DeepSeek-R1-32B）
#: 三条 case 全部因为「host_snr 未采集」弃权——而 C14 本就说明该字段常态缺失。
#: 模型把「遥测不全」当成了「证据不足」。v2 把判据改成「可用证据能否区分候选根因」。
PROMPT_TEMPLATE_VERSION = "rca-constrained-reasoning-v6"
SOP_VERSION = "learned-sop-advisory-v2"

ROOT_CAUSE_DEFINITIONS = {
    "L1": "400G 端口一侧的设备或端口根因",
    "L2": "200G 端口一侧的设备或端口根因",
    "fiber": "L1 与 L2 之间的光纤 / 链路介质根因",
}

SYSTEM_PREAMBLE = """你是光链路故障定界专家。你的任务是在给定的物理约束内，
依据给定证据判断根因，或者在证据不足时明确弃权。

硬性要求：
1. 只能引用「可用证据」清单里列出的 token。不得引用清单之外的任何证据，
   也不得描述清单里没有的观测。编造证据会导致整次回答被判为不合规。
2. 只能引用「物理约束」清单里的约束编号。
3. 每一步推理都必须至少引用一条证据或一条约束，不允许凭空断言。
4. 标注为「待专家审核」的约束可以参考，但不能作为唯一定论依据。
5. 「训练集归纳 SOP」是训练历史的统计先验，不是当前 case 的物理证据。
   它可以帮助安排检查顺序，但不能被写入 `cited_evidence`，也不能单独支撑结论。
6. 不要为了填字段而引用约束。`effect=neutral` 且 `target=空字符串` 的约束只是
   防止误读数据的护栏，严禁写入 `cited_constraints`；遵守其文字要求即可。
7. 只有契约明确允许 `support` 或 `exclude` 的约束才可写入 `cited_constraints`，
   且该步的 effect、target 和证据 token 前缀必须同时满足该约束的结构化引用契约。
8. 约束编号必须从清单中逐字完整复制，例如 `C11_media_snr_floor`，禁止缩写为 `C11`。
9. 不引用约束的推理步骤必须明确写 `"cited_constraints": []`。

判据（决定给结论还是弃权）：

- 判断标准是**可用证据能不能把候选根因区分开**，不是遥测采全没采全。
- 遥测不完整是这批数据的常态，不是弃权理由。「未采集字段」清单只用来填
  `missing_information`，说明补采什么能提高把握；它本身不构成弃权依据。
  约束 C14 已经写明 host_snr 大多数 case 不采集——若因为它缺失就弃权，
  那就是对所有 case 都弃权，等于没有判断。
- 可用证据明确指向某一个根因时，就给出该根因，并用 `confidence` 表达把握大小。
  把握不足应当体现为低 confidence，而不是直接弃权。
- 只有在下面两种情况才填 `abstain`：
  (a) 可用证据与两个及以上候选根因同样吻合，没有任何证据能把它们区分开；
  (b) 可用证据与所有候选根因都矛盾，或证据本身自相矛盾。
- 弃权本身不是失败，给出没有证据支撑的结论才是；但在证据足以区分时弃权，
  同样是一次错误的判断。
"""

OUTPUT_INSTRUCTION = """只输出一个 JSON 对象，结构如下：
{
  "steps": [
    {
      "claim": "这一步的断言",
      "cited_evidence": ["引用的证据 token"],
      "cited_constraints": ["引用的约束编号"],
      "effect": "support | exclude | neutral",
      "target": "L1 | L2 | fiber | \\"\\""
    }
  ],
  "verdict": "L1 | L2 | fiber | abstain",
  "confidence": 0.0,
  "missing_information": ["还需要补采什么才能提高把握"]
}"""


def _evidence_section(request: Any) -> Dict[str, Any]:
    branch_explanations = {
        "N5a": "证据与历史 case 完全匹配，但历史标签不纯；历史投票只作上下文，需用物理证据仲裁。",
        "N5b": "证据与历史 case 部分匹配；缺关键证据或历史候选冲突，需用物理证据仲裁。",
        "N5c": "历史证据图里没有足够相似的 case，不能复用历史结论，只能依据物理约束推理。",
    }
    return {
        "case_id": request.case_id,
        "路由分支": request.branch,
        "路由原因": request.routing_reason,
        "可用证据": list(request.evidence_tokens),
        "遥测完整性": request.telemetry_status,
        "未采集字段（常态，仅供填写 missing_information，不是弃权理由）":
            list(request.missing_fields),
        "历史最高相似度": request.nearest_similarity,
        "历史匹配说明": branch_explanations.get(request.branch, branch_explanations["N5c"]),
        "历史候选 case": list(request.historical_case_ids),
        "历史候选标签分布（仅作上下文，不能替代物理证据）":
            dict(request.historical_label_distribution),
        "训练集归纳 SOP（仅作检查路径与统计先验，不能作为 cited_evidence）":
            getattr(request, "sop_prediction", None),
    }


def build_prompt(
    request: Any,
    *,
    library: ConstraintLibrary = CONSTRAINT_LIBRARY,
    retry_feedback: str = "",
) -> str:
    """渲染 N5c 的推理 prompt。

    `retry_feedback` 是上一轮的约束校验失败原因。它必须被放在最前面且写明是
    「上一次回答的问题」，否则模型会把它当成新的证据。
    """
    already_excluded = [
        {"根因": item.root_cause, "依据约束": item.constraint_id, "原因": item.reason}
        for item in request.exclusions
    ]
    payload = {
        "任务": f"光链路根因三分类定界（{request.branch} 分支）",
        "根因定义": ROOT_CAUSE_DEFINITIONS,
        "本 case 证据": _evidence_section(request),
        "已由确定性物理排除排掉的根因": already_excluded,
        "可选根因": list(request.candidate_root_causes),
    }

    constraints = [library.get(constraint_id) for constraint_id in request.constraint_ids]
    sections = [SYSTEM_PREAMBLE]
    if retry_feedback:
        sections.append(
            "上一次回答未通过物理约束校验，问题如下。请针对这些问题重写，"
            "不要重复同样的错误：\n" + retry_feedback
        )
    sections.append("物理约束（按 排除 -> 禁止推断 -> 恒等关系 -> 倾向性 排序）：\n"
                    + render_prompt_block(constraints=constraints))
    sections.append(json.dumps(payload, ensure_ascii=False, indent=2))
    sections.append(OUTPUT_INSTRUCTION)
    return "\n\n".join(sections)


def prompt_template_hash() -> str:
    """模板内容指纹；版本号忘记升级时，代码变化仍会让实验 manifest 改变。"""
    payload = "\n".join(
        (
            PROMPT_TEMPLATE_VERSION,
            SOP_VERSION,
            SYSTEM_PREAMBLE,
            OUTPUT_INSTRUCTION,
            inspect.getsource(_evidence_section),
            inspect.getsource(build_prompt),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
