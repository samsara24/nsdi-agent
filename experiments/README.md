# RCA 实验归档规范

每一次 Loop 实验必须在本目录下新建一个目录：

```text
experiments/<YYYYMMDD>_<short-name>/
```

该目录至少包含：

- `report.html`：主实验报告，必须从当前主流程图出发，标注本轮改动点。
- `experiment_manifest.json`：实验配置、版本、改动类别和 N8 冻结状态。
- `summary.json`：聚合指标、分支统计、正确 case 归纳、bad case 分类统计。
- `case_analysis.json`：逐 case 分析，正确 case 与 bad case 都要保留。
- `bad_cases.json`：本轮所有测试 bad case 的错因、失败步骤和下一步动作。
- `label_suspects.json`：疑似 label 问题的 case，后续实验必须继续保留。
- `irreducible_cases.json`：当前证据、训练集、专家规则和 SOP 都无法安全提升的 case，后续实验保留但不继续围绕它刷指标。

## 实验允许改动类别

每次实验只能声明并验证以下类别之一或少量组合：

1. `evidence_graph`：证据图 schema、链路边、约束节点、匹配逻辑或历史链路摘要。
2. `threshold_or_routing`：N4 路由阈值、M9 降级门限、覆盖率/风险工作点。
3. `llm_prompt`：N5b/N5c prompt、专家 SOP 注入方式、LLM 输出协议。
4. `bug_fix`：代码实现与既定设计不一致的修复。

禁止把“提升强制三分类 accuracy / lift”作为唯一目标。所有结论必须回到测试 bad case：哪里错、为什么错、这类错能不能用当前证据安全修。

## HTML 报告必须回答

- 当前主流程图是什么，本轮在哪里做了调整。
- 当前证据图是什么：节点、边、匹配方式、历史链路摘要。
- 当前物理约束是什么：纯物理、量测契约、统计先验分别有哪些版本。
- SOP 是否变化：专家 SOP 与 learned SOP / 数值树各自版本和使用边界。
- 正确 case：按分支统计，说明哪些步骤做得好。
- Bad case：逐条说明失败步骤、错因、是否疑似 label 问题、是否 irreducible。
- 下一轮只应关注哪些可修复 bad case，不再反复优化哪些 irreducible case。
