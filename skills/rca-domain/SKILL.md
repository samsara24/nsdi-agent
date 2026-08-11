---
name: rca-domain
description: 光链路 RCA 的领域边界、标签语义和量测禁区。
---

# RCA Domain

- `L1`：400G 端口或其设备侧根因。
- `L2`：200G 端口或其设备侧根因。
- `fiber`：L1 与 L2 之间的光纤 / 链路介质根因。

## 数据边界

- `rca_v2_l2fixed` 是 RCA v2 新实验数据源；legacy organized 126/85 仅保留回归锚点。
- L1 是 400G、L2 是 200G。lane 数可以不同；在厂商确认 lane 对应前，禁止跨端按 lane 编号计算绝对链路损耗。
- `serdes_snr` 量纲未知，只能作为有效 / 失效二值状态，不得按 dB SNR 解释。
