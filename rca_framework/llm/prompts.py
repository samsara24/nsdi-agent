"""M8 固定 prompt 模板。

三条原则：

1. **prompt 不逐 case 手写。** 全部由 `DiagnosisRequest` 与约束库渲染，
   模板本身有版本号并进 `run_manifest.json`。手写 prompt 会让实验不可复现。
2. **约束按类型排序注入：exclusion -> caveat -> invariant -> indicator。**
   先给能排除的，再给不许推的，最后才给提高可能性的。理由见 T2：
   这套数据里「倾向性证据」很容易把模型推向多数类，先做排除可以在加权之前砍掉不可能的选项。
   这个顺序由 `constraints.library.render_prompt_block` 保证，有测试锁定。
3. **每个 case 必须给三分类候选。** 证据不足不再用 abstain 表达，而是通过
   多维置信度把低证据完整度、低物理合规性和推理缺口显式暴露给 N6 阈值门禁。
"""

from __future__ import annotations

import json
import hashlib
import inspect
from typing import Any, Dict, Optional, Sequence

from ..constraints.library import CONSTRAINT_LIBRARY, ConstraintLibrary, render_prompt_block
from ..constraints.measurement import MEASUREMENT_CONTRACT_LIBRARY
from ..constraints.physics import PHYSICS_LIBRARY
from ..sop.expert_sop import EXPERT_SOP_VERSION, render_expert_sop_prompt_block
from ..types import ROOT_CAUSES
from .confidence_rubric import CONFIDENCE_RUBRIC
from .prompt_templates.diagnose import build_diagnose_prompt, diagnose_prompt_version_for


#: v2 改写了弃权判据。v1 只说「证据不足就弃权」，真机实测（DeepSeek-R1-32B）
#: 三条 case 全部因为「host_snr 未采集」弃权——而 C14 本就说明该字段常态缺失。
#: 模型把「遥测不全」当成了「证据不足」。v2 把判据改成「可用证据能否区分候选根因」。
#: v7 加入归因方向表。迭代 2 的失败分析显示，73 条 LLM 回答里 58 条被约束校验打回，
#: 其中最集中的一类是**归因方向反了**：模型看到「L1 侧收光异常」就把根因写成 L1，
#: 而接收侧看到的光是对端发出的，本端发送器根本不在这条光路上。
#: 这不是 prompt 措辞问题——v6 的约束清单里确实写了 C16，但它混在 20 多条约束中间，
#: 模型要先自己意识到「该用方向类约束」才会去读。v7 把方向表提到系统级硬规则，
#: 让它在读证据之前就已经知道每一类观测指向哪一端。
LEGACY_PROMPT_TEMPLATE_VERSION = "rca-dual-sop-multidim-v14-full-step-ids"
FILTERED_RULE_PROMPT_TEMPLATE_VERSION = "filtered-rule-local-remote-v1"
# Existing entrypoints import this name directly; keep it on the legacy contract.
PROMPT_TEMPLATE_VERSION = LEGACY_PROMPT_TEMPLATE_VERSION
SOP_VERSION = "expert-sop-n5c-v1+learned-sop-advisory-v2"
FILTERED_RULE_TOPOLOGY_CONTRACT = "filtered-rule-topology-v1"

ROOT_CAUSE_DEFINITIONS = {
    "L1": "400G 端口一侧的设备或端口根因",
    "L2": "200G 端口一侧的设备或端口根因",
    "fiber": "L1 与 L2 之间的光纤 / 链路介质根因",
}

#: 现网专家的归因方向表（`docs/EXPERT_EXPERIENCE.md` §5.3 / §7）。
#:
#: 它先于任何统计证据成立，因为它来自链路的物理结构而不是本数据集的相关性：
#: 一侧的接收类读数度量的是**对端发出、穿过光纤后到达本端**的光，
#: 而发送类与电口读数度量的是**本端自己产生**的信号。
#: 把它放进系统级 preamble 而不是约束清单，是因为它是读证据的**前提**：
#: 模型必须在解释任何一个 token 之前就知道这个 token 约束的是哪一端。
ATTRIBUTION_DIRECTION_RULE = """归因方向（先于一切证据解释，任何一步推理都不得违反）：

光链路是双向的，一端看到的现象未必由这一端造成。判断「症状在哪一端」之后，
必须先按下表把它翻译成「根因在哪一端」，再去找支持它的约束：

| 观测到的异常 | 度量的是什么 | 根因指向 |
| --- | --- | --- |
| rxpower（接收光功率）异常 | 对端发出、穿过光纤到达本端的光 | **对端** |
| media_snr（介质侧信噪比）异常 | 同上，收到的光的质量 | **对端** |
| 上面两项 + serdes_snr 在同一侧同时异常 | 整条接收通道都拿到坏信号 | **对端**（证据更强） |
| txpower（发送光功率）异常 | 本端自己发出的光 | **本端** |
| host_snr（主机侧信噪比）异常 | 本端电口进来的信号 | **本端** |
| serdes_snr 单独异常 | 本端 SerDes 的信号质量 | **本端** |

因此：「L1 侧收光低」支持的是 **L2**，不是 L1；「L2 侧收光低」支持的是 **L1**，不是 L2。
把接收侧症状归给报症状的那一端，是本任务上最常见也最严重的错误。

两端都有异常时按专家优先级仲裁，数值小的先赢：
发送侧断光(0) > 三项组合异常(1) > host_snr(2) > serdes_snr(3) > media_snr(4) > rxpower(5) > 发送侧非断光异常(6)。
两端优先级相同但指向不同时才考虑光纤。

「该侧一切正常」是合法且有用的观察：把它写成 `effect=neutral`、`target=""` 的一步，
不要为了凑证据把正常读数说成异常。
"""


SYSTEM_PREAMBLE = """你是光链路故障定界专家。你的任务是在给定的物理约束内，
依据给定证据判断最可能根因，并为后续阈值门禁输出多维置信度。

""" + ATTRIBUTION_DIRECTION_RULE + """
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
10. 每一步必须从载荷中逐字复制完整 `sop_step_id`，并按
    `Q0_validate_measurements` → `P_apply_physical_boundaries` → `R_expand_directional_chain` →
    `L_apply_stable_learned_ranges` → `D_select_or_request_evidence` 的顺序引用已执行 SOP；
    禁止使用 `Q0`、`P`、`R`、`L`、`D` 缩写；
    只能使用载荷中声明的谓词和阈值，禁止临时发明或修改连续阈值。
11. 每一步必须至少引用一个当前 case 的 `cited_evidence` 和一个已声明的 `cited_predicates`；
    引用不存在的证据或谓词会使整条结论作废。

判据（强制三选一 + 多维低置信表达）：

- `verdict` 必须在 L1 / L2 / fiber 中三选一，禁止输出 abstain、unknown、insufficient_evidence。
- 判断标准是**可用证据更偏向哪个根因**，不是遥测采全没采全。证据不足时仍选最可能的根因，
  但必须把 `evidence_completeness` 和/或 `reasoning_completeness` 打低分。
- 遥测不完整是这批数据的常态。「未采集字段」清单只用来填 `missing_information`，
  说明补采什么能提高把握；它本身不替代物理证据。
- 关于 fiber（C20）：现有两端遥测通常无法唯一识别介质根因，端点根因是默认解释。
  只有同时具备「两端均已发光」和「同一 lane 双向对称路径丢失」两条证据时，才可把 fiber 写成
  `verdict`。只有单向「本端发光正常、对端收不到」时，target 取对端端点，不取 fiber。
  若在缺少双向现场证据的情况下仍判 fiber，`physical_compliance` 必须 <= 0.3，
  并在 `missing_information` 中请求 OTDR / 端面镜检 / 双向功率标定 / 换纤复测。
- `verdict` 必须与 steps 自洽：把所有 `effect=support` 的 target 汇总、减去 `effect=exclude`
  的 target，`verdict` 取得票最高的那一个。若想给的结论与推理链汇总不一致，
  先补写能支撑它的 support 步骤，不要直接输出与自己推理链矛盾的结论。

""" + CONFIDENCE_RUBRIC + """
"""

OUTPUT_INSTRUCTION = """只输出一个 JSON 对象，结构如下：
{
  "steps": [
    {
      "sop_step_id": "P_apply_physical_boundaries",
      "cited_predicates": ["载荷中声明的谓词 ID"],
      "claim": "这一步的断言",
      "cited_evidence": ["引用的证据 token"],
      "cited_constraints": ["引用的约束编号"],
      "effect": "support | exclude | neutral",
      "target": "L1 | L2 | fiber | \\"\\""
    }
  ],
  "verdict": "L1 | L2 | fiber",
  "confidence": 0.0,
  "confidence_breakdown": {
    "evidence_completeness": 0.0,
    "physical_compliance": 0.0,
    "reasoning_completeness": 0.0,
    "history_similarity": 0.0
  },
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
        "原始数值、单位与 lane 数": getattr(request, "raw_measurements", {}),
        "S_feature": getattr(request, "feature_similarity", 0.0),
        "S_graph": getattr(request, "graph_similarity", 0.0),
        "五层证据路径": list(getattr(request, "evidence_paths", ())),
        "对立历史 case": list(getattr(request, "opposing_historical_cases", ())),
        "最大特征差异": list(getattr(request, "largest_differences", ())),
        "关键缺失证据": list(getattr(request, "critical_missing_evidence", ())),
        "允许使用的谓词与阈值来源": list(getattr(request, "declared_predicates", ())),
        "已执行 SOP（必须按顺序引用 sop_step_id）": list(getattr(request, "sop_trace", ())),
        "确定性 SOP 候选集": list(getattr(request, "sop_candidates", ())),
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
    # Filtered-rule cases have explicit local/remote topology. Legacy N5c cases use
    # the same pure-physics renderer with the original 400G/200G label semantics.
    if _is_filtered_rule_request(request):
        return build_diagnose_prompt(
            request,
            retry_feedback=retry_feedback,
            profile="filtered_rule_v1",
        )
    if getattr(request, "branch", "") == "N5c":
        return build_diagnose_prompt(request, retry_feedback=retry_feedback, profile="legacy")

    # Legacy rendering remains active for existing organized/l2fixed entrypoints.
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


def _is_filtered_rule_request(request: Any) -> bool:
    topology_context = getattr(request, "topology_context", {}) or {}
    return topology_context.get("contract_version") == FILTERED_RULE_TOPOLOGY_CONTRACT


def prompt_template_version_for(request: Any = None, *, profile: str = "") -> str:
    if profile == "filtered_rule_v1" or (request is not None and _is_filtered_rule_request(request)):
        return FILTERED_RULE_PROMPT_TEMPLATE_VERSION
    return LEGACY_PROMPT_TEMPLATE_VERSION


def prompt_template_hash(profile: str = "legacy") -> str:
    """模板内容指纹；版本号忘记升级时，代码变化仍会让实验 manifest 改变。"""
    prompt_version = prompt_template_version_for(profile=profile)
    diagnose_version = diagnose_prompt_version_for(profile)
    payload = "\n".join((
        prompt_version,
        diagnose_version,
        SOP_VERSION,
        EXPERT_SOP_VERSION,
        PHYSICS_LIBRARY.content_hash(),
        MEASUREMENT_CONTRACT_LIBRARY.content_hash(),
        CONFIDENCE_RUBRIC,
        render_expert_sop_prompt_block(),
        inspect.getsource(build_diagnose_prompt),
        inspect.getsource(build_prompt),
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
