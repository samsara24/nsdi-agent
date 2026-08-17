"""Render RCA project skills from structured code/artifact contracts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca_framework.constraints.library import CONSTRAINT_LIBRARY  # noqa: E402
from rca_framework.evidence_graph import EVIDENCE_GRAPH_V2_SCHEMA  # noqa: E402
from rca_framework.features.dictionary import FEATURE_DICTIONARY_V2  # noqa: E402
from rca_framework.sop import EXPERT_SOP_VERSION, LEARNED_SOP_VERSION, expert_sop_hash  # noqa: E402
from scripts.render_constraint_skill import render as render_constraints  # noqa: E402


SKILL_DIR = Path("skills")


def _front(name: str, description: str) -> str:
    return f"---\nname: {name}\ndescription: {description}\n---\n\n"


def render_domain() -> str:
    return _front("rca-domain", "光链路 RCA 的领域边界、标签语义和量测禁区。") + "\n".join([
        "# RCA Domain",
        "",
        "- `L1`：400G 端口或其设备侧根因。",
        "- `L2`：200G 端口或其设备侧根因。",
        "- `fiber`：L1 与 L2 之间的光纤 / 链路介质根因。",
        "",
        "## 数据边界",
        "",
        "- `rca_v2_l2fixed` 是 RCA v2 新实验数据源；legacy organized 126/85 仅保留回归锚点。",
        "- L1 是 400G、L2 是 200G。lane 数可以不同；在厂商确认 lane 对应前，禁止跨端按 lane 编号计算绝对链路损耗。",
        "- `serdes_snr` 量纲未知，只能作为有效 / 失效二值状态，不得按 dB SNR 解释。",
        "",
    ])


def render_sop() -> str:
    return _front("rca-sop", "RCA v2 的专家 SOP 与 learned SOP 使用边界。") + "\n".join([
        "# RCA SOP",
        "",
        f"当前专家 SOP 版本：`{EXPERT_SOP_VERSION}`，hash `{expert_sop_hash()}`。",
        f"当前 learned SOP 版本：`{LEARNED_SOP_VERSION}`。",
        "",
        "专家 SOP 是 N5c 冷启动分支的检查顺序，用于约束 LLM 逐步推理校验。",
        "learned SOP 是从训练集标签归纳得到的浅层可解释决策树，不是专家手写 SOP。",
        "使用 learned SOP 时必须同时检查叶节点支持数、叶子纯度和 Wilson 下界；低支持或混合叶必须补采或转人工。",
        "",
        "## 使用规则",
        "",
        "1. N5a/N5b 不注入完整专家 SOP；只有 N5c 冷启动注入专家 SOP。",
        "2. learned SOP / 数值树只能作为统计先验或报告字段，不能默认进入 M9 自动终裁。",
        "3. test split 只做最终评估，不能反向修改树、约束、SOP 或特征。",
        "4. learned SOP 不得覆盖确定性物理排除，也不得把待专家确认的统计关系写成物理事实。",
        "",
    ])


def render_graph() -> str:
    return _front("rca-evidence-graph", "RCA v2 证据图 schema、节点边和回灌边界。") + "\n".join([
        "# Evidence Graph",
        "",
        f"当前 schema：`{EVIDENCE_GRAPH_V2_SCHEMA}`。",
        f"当前 v2 特征字典：`{FEATURE_DICTIONARY_V2.version}`，hash `{FEATURE_DICTIONARY_V2.content_hash()}`。",
        "",
        "## 双层结构",
        "",
        "- 全局 case-token 图：用于 N3 历史检索与 IDF 加权 Jaccard。",
        "- per-case 诊断图：Observation / FeatureToken / ConstraintCheck / SOPStep / Outcome。",
        "",
        "## 回灌规则",
        "",
        "- 自动推理结果只能写入实验 artifact；只有人工确认的 case 才能回灌到证据图。",
        "- 回灌必须记录 `confirmed_by`、SOP 版本、约束库版本和 graph content hash。",
        "",
    ])


def render_workflow() -> str:
    return _front("rca-workflow", "RCA v2 N1-N8 主流程、降级策略和实验门禁。") + "\n".join([
        "# RCA Workflow",
        "",
        "1. N1：构造 EvidencePack，结构性剥离标签。",
        "2. N2：抽取可解释 token，连续阈值只从 train split 拟合。",
        "3. N3：证据图检索 Top-N，输出相似度、覆盖率、缺失和冲突证据。",
        "4. N4：用当前数据集 train-LOO 重新标定路由，不沿用旧 70% 阈值。",
        "5. N5：N5a 复用纯历史链，N5b 用物理约束判关键证据并仲裁，N5c 走专家 SOP + 约束 LLM。",
        "6. N6：正式默认只接受 branch 候选；expert / learned SOP 只能显式消融或作报告字段。",
        "7. N7：生成含主流程图、调整点、根因、证据链、SOP 路径、置信来源和逐 bad case 分析的报告。",
        "8. N8：本阶段冻结；只保留人工确认回灌语义，不用测试 bad case 自动更新知识。",
        "",
        "## Loop 实验门禁",
        "",
        "- 每轮实验必须先说明遵循 `docs/个人整体思路.md`，并从测试 bad case 出发提出假设。",
        "- 允许的核心调整只有：证据图约束 / schema、阈值或路由、大模型 prompt、代码 bug fix。",
        "- 每轮实验必须归档到 `experiments/<YYYYMMDD>_<short-name>/`，并生成 `report.html`。",
        "- `report.html` 必须展示当前主流程图，并标注本轮调整了哪里。",
        "- 报告必须记录当前证据图、物理约束、SOP 版本、prompt 版本、阈值和 M9 candidate order。",
        "- 正确 case 按分支和做对的步骤归纳；bad case 必须逐条分析失败步骤、错因和下一步动作。",
        "- 疑似标签问题写入 `label_suspects.json`；当前不可安全提升的 case 写入 `irreducible_cases.json`，后续保留但不继续围绕它刷指标。",
        "",
    ])


def renders() -> Dict[Path, str]:
    return {
        SKILL_DIR / "rca-domain" / "SKILL.md": render_domain(),
        SKILL_DIR / "rca-constraints" / "SKILL.md": render_constraints(CONSTRAINT_LIBRARY),
        SKILL_DIR / "rca-sop" / "SKILL.md": render_sop(),
        SKILL_DIR / "rca-evidence-graph" / "SKILL.md": render_graph(),
        SKILL_DIR / "rca-workflow" / "SKILL.md": render_workflow(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    stale = []
    for path, content in renders().items():
        if args.check:
            current = path.read_text(encoding="utf-8") if path.exists() else ""
            if current != content:
                stale.append(str(path))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            print(f"wrote {path}")
    if stale:
        raise SystemExit("stale skills: " + ", ".join(stale))
    if args.check:
        print("RCA skills are up to date")


if __name__ == "__main__":
    main()
