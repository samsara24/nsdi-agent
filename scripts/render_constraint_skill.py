"""从 M5 约束库生成 `skills/rca-constraints/SKILL.md`。

SKILL.md 不手写，只从 `constraints/library.py` 渲染。这样约束门限只有一处定义，
prompt 文本与代码常量不可能漂移。`tests/test_constraint_library.py` 会重新渲染并
与磁盘上的文件比对，文件过期时测试直接失败。

用法::

    python scripts/render_constraint_skill.py            # 写入 skills/rca-constraints/SKILL.md
    python scripts/render_constraint_skill.py --check    # 只校验是否与代码一致
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rca_framework.constraints.library import (  # noqa: E402
    CONSTRAINT_LIBRARY,
    ConstraintLibrary,
    render_prompt_block,
)


SKILL_PATH = Path("skills/rca-constraints/SKILL.md")

FRONT_MATTER = """---
name: rca-constraints
description: 光链路 RCA 的物理约束库（M5）。在 N5b 补证据与 N5c 通用排障推理时注入，用于约束 LLM 的每一步推断。本文件由 scripts/render_constraint_skill.py 从 rca_framework/constraints/library.py 自动生成，不要手工编辑。
---
"""


def render(library: ConstraintLibrary = CONSTRAINT_LIBRARY) -> str:
    kinds = {
        "exclusion": "排除条件",
        "caveat": "禁止推断",
        "invariant": "物理恒等",
        "indicator": "倾向性线索",
    }
    lines = [
        FRONT_MATTER,
        "# 光模块物理约束库",
        "",
        f"版本 `{library.version}`，内容指纹 `{library.content_hash()}`，共 {len(library.constraints)} 条。",
        f"`measured` 类参数的实测口径：{library.measured_on}。",
        "",
        "## 使用方式",
        "",
        "1. N5c（低匹配 / 未见模式）必须注入全部约束。",
        "2. N5b（部分匹配）只在需要补证据或仲裁冲突时注入相关类别。",
        "3. 推理的每一步都要能指到具体的约束 ID；指不到就说明该步没有物理依据。",
        "4. 执行顺序固定：先用排除条件砍掉不可能的根因，再看禁止推断避免走进死胡同，",
        "   最后才用倾向性线索排序剩余候选。",
        "",
        "## 约束清单",
        "",
        "| ID | 类别 | 类型 | 断言 | 参数来源 | 审核状态 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in library.constraints:
        parameters = "；".join(f"{name}={value}" for name, value in item.parameters) or "无参数"
        lines.append(
            f"| `{item.constraint_id}` | {item.category} | {kinds[item.kind]} | {item.title} "
            f"| {item.provenance}（{parameters}） | {item.review_status} |"
        )

    lines += ["", "## 逐条说明", ""]
    for item in library.constraints:
        lines += [
            f"### {item.constraint_id} — {item.title}",
            "",
            f"- **物理依据**：{item.physical_statement}",
            f"- **形式表达**：`{item.formal_expression}`",
            f"- **实测证据**：{item.measured_evidence}",
            f"- **诊断用法**：{item.diagnostic_use}",
            "",
        ]

    lines += [
        "## 注入 prompt 的文本块",
        "",
        "以下内容由 `render_prompt_block()` 产出，是真正进入 prompt 的原文。",
        "",
        "```text",
        render_prompt_block(library).rstrip(),
        "```",
        "",
        "## 待办",
        "",
        f"- 全部 {len(library.constraints)} 条均为 `pending_expert_review`，需夏思博逐条确认后改为 `approved`。",
        "- `measured` 类参数绑定当前数据集切分，合并数据集到位后必须重测。",
        "- `C12` 与 `C13` 是数据质量问题，需要向厂商确认 lane 编号对应关系与 `serdes_snr` 量纲。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只校验文件是否与代码一致")
    parser.add_argument("--path", type=Path, default=SKILL_PATH)
    args = parser.parse_args()

    content = render()
    if args.check:
        current = args.path.read_text(encoding="utf-8") if args.path.exists() else ""
        if current != content:
            print(f"{args.path} 与约束库不一致，请运行 python scripts/render_constraint_skill.py")
            raise SystemExit(1)
        print(f"{args.path} 与约束库一致")
        return

    args.path.parent.mkdir(parents=True, exist_ok=True)
    args.path.write_text(content, encoding="utf-8")
    print(f"wrote {args.path}")


if __name__ == "__main__":
    main()
