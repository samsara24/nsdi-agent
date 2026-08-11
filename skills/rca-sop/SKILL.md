---
name: rca-sop
description: RCA v2 的 learned SOP 使用边界和决策树契约。
---

# Learned SOP

当前 SOP 版本：`learned-sop-v1`。

该 SOP 是从训练集标签归纳得到的浅层可解释决策树，不是专家手写 SOP。
使用时必须同时检查叶节点支持数、叶子纯度和 Wilson 下界；低支持或混合叶必须补采或转人工。

## 使用规则

1. 只允许用 manifest train split 学习树结构和剪枝参数。
2. test split 只做最终评估，不能反向修改树、约束或特征。
3. 每条路径必须记录 `present:<token>` / `absent:<token>`，报告中展示完整路径。
4. learned SOP 不得覆盖确定性物理排除，也不得把待专家确认的统计关系写成物理事实。
