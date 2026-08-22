# Filtered-rule temporal split v1

本目录是项目唯一活动数据契约，不包含模型输出。

## 划分

- `train/`：两个来源中 2025-06 至 2025-09 的 case，共 124 条。
- `test/all_data/`：`all_data` 其余月份，共 417 条。
- `test/rule1_channel_not_4/`：`rule1_channel_not_4` 其余月份，共 67 条。

训练集统一建模，两个测试集使用同一只读知识包分别评估。

## 标签与端点

- `all_data`：`l1 -> L1（本端 400G）`，`l2 -> L2（对端 200G）`。
- `rule1_channel_not_4`：`l3 -> L1（本端 400G）`，`l4 -> L2（对端 400G）`。
- `fiber -> fiber`。

`L1/L2` 的根因语义始终是本端/对端；端口速率作为 topology context 独立保存。

## Logical lane contract

- `all_data` 的光学指标为 4x4 个逻辑 lane。
- `rule1_channel_not_4` 的光学指标主要为 8x8 个逻辑 lane，SerDes 为 4x4。
- 两个来源的 `transmission` 均按同号逻辑 lane 计算远端 RX 与本端 TX 的差值。
- 同号 optical lane 可用于断光状态、双向状态和 case 内相对离群证据。
- 不允许把两端功率差解释为绝对链路损耗。
- 不允许把 4-lane SerDes 与 8-lane optical 指标一一配对。

每条 case 的 `_dataset_contract` 保存来源、端口速率、topology ID、lane alignment 规则、原始标签与源文件哈希。

## Expert label

上一版人工标注通过 12 组核心遥测字段精确指纹映射到 `all_data`：匹配 49 条，实际修正 27 条。没有精确匹配的 case 不做推断式改标。

## 复核

```bash
python3 scripts/prepare_filtered_rule_temporal_split.py --check
```
