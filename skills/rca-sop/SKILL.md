---
name: rca-sop
description: RCA v2 的专家 SOP 与 learned SOP 使用边界。
---

# RCA SOP

当前专家 SOP 版本：`expert-sop-playbook-v2`，hash `55da6164d312bdf7`。
当前 learned SOP 版本：`learned-sop-v1`。

专家 SOP 是 N5c 冷启动分支的检查顺序，用于约束 LLM 逐步推理校验。
learned SOP 是从训练集标签归纳得到的浅层可解释决策树，不是专家手写 SOP。
使用 learned SOP 时必须同时检查叶节点支持数、叶子纯度和 Wilson 下界；低支持或混合叶必须补采或转人工。

## 使用规则

1. N5a/N5b 不注入完整专家 SOP；只有 N5c 冷启动注入专家 SOP。
2. learned SOP / 数值树只能作为统计先验或报告字段，不能默认进入 M9 自动终裁。
3. test split 只做最终评估，不能反向修改树、约束、SOP 或特征。
4. learned SOP 不得覆盖确定性物理排除，也不得把待专家确认的统计关系写成物理事实。
