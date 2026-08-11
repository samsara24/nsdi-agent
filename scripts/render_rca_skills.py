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
from rca_framework.sop import LEARNED_SOP_VERSION  # noqa: E402
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
    return _front("rca-sop", "RCA v2 的 learned SOP 使用边界和决策树契约。") + "\n".join([
        "# Learned SOP",
        "",
        f"当前 SOP 版本：`{LEARNED_SOP_VERSION}`。",
        "",
        "该 SOP 是从训练集标签归纳得到的浅层可解释决策树，不是专家手写 SOP。",
        "使用时必须同时检查叶节点支持数、叶子纯度和 Wilson 下界；低支持或混合叶必须补采或转人工。",
        "",
        "## 使用规则",
        "",
        "1. 只允许用 manifest train split 学习树结构和剪枝参数。",
        "2. test split 只做最终评估，不能反向修改树、约束或特征。",
        "3. 每条路径必须记录 `present:<token>` / `absent:<token>`，报告中展示完整路径。",
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
        "5. N5：N5a 复用纯历史链，N5b 补采/仲裁，N5c 走约束 + learned SOP。",
        "6. N6：按历史覆盖率、SOP 叶子校准、约束合规、证据完整度和推导缺口决定 final / request_evidence / human_review。",
        "7. N7：生成含根因、证据链、SOP 路径和置信来源的报告。",
        "8. N8：只回灌人工确认结果。",
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
