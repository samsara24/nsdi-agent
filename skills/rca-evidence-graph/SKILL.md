---
name: rca-evidence-graph
description: RCA v2 证据图 schema、节点边和回灌边界。
---

# Evidence Graph

当前 schema：`evidence-graph-v2`。
当前 v2 特征字典：`feature-dictionary-v2`，hash `78bbdbbf601fe29e`。

## 双层结构

- 全局 case-token 图：用于 N3 历史检索与 IDF 加权 Jaccard。
- per-case 诊断图：Observation / FeatureToken / ConstraintCheck / SOPStep / Outcome。

## 回灌规则

- 自动推理结果只能写入实验 artifact；只有人工确认的 case 才能回灌到证据图。
- 回灌必须记录 `confirmed_by`、SOP 版本、约束库版本和 graph content hash。
