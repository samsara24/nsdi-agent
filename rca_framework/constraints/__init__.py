"""M5 物理约束库。

约束是「光模块物理属性」这条论文动机的落点：它们不是从数据里学出来的统计规律，
而是器件与链路本身成立的关系。约束库同时承担两个用途：给 N5c 的 LLM prompt 提供
可注入的物理先验（M5），以及给 M7 校验器提供可执行断言。
"""

from .library import (
    CONSTRAINT_LIBRARY,
    CONSTRAINT_LIBRARY_VERSION,
    Constraint,
    ConstraintLibrary,
    render_prompt_block,
)

__all__ = [
    "CONSTRAINT_LIBRARY",
    "CONSTRAINT_LIBRARY_VERSION",
    "Constraint",
    "ConstraintLibrary",
    "render_prompt_block",
]
