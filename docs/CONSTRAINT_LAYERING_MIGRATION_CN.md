# 物理约束库三层迁移映射

本文记录 `constraint-library-v6` 中 26 条旧约束在重构后的归属。目标是把“物理事实”“量测契约”和“训练集决策知识”拆开，避免把训练集分位数和叶节点统计当作物理规则注入 LLM。

## 三层定义

- **L1 纯物理约束**：器件、链路拓扑或信号方向决定的关系，不依赖训练集标签分布。允许引用器件规格常量，例如断光哨兵、模块温度规格、电压规格。
- **L2 量测契约**：这份数据能不能被这样解释。命中后用于否决某条推理或转补采，不作为支持根因的证据。
- **L3 决策树知识**：从训练集分位数、工程阈值命中分布或 Wilson 下界得到的分支与叶节点统计。它描述“在训练数据中这个条件对应什么结论”，不是物理约束。

## 逐条迁移表

| 旧 ID | 新归属 | 迁移说明 |
| --- | --- | --- |
| C1_bias_zero_means_laser_off | L1 | 保留“bias 为 0 表示激光器未点亮”的物理内核；训练集共现统计只作审计备注。 |
| C2_bias_healthy_band | L3 | `7.2-7.8 mA` 是训练集实测健康带，应作为数值树候选分裂点，不进入物理库。 |
| C3_temperature_operating_range | L1 | `0-70 °C` 是商用模块工作规格，可作为设备规格物理约束。 |
| C4_voltage_nominal_band | L1 | `3.3 V ±5%` 是供电规格，可作为设备规格物理约束。 |
| C5_tx_power_range | L1 + L3 | “tx 只有有光/无光两态、哨兵表示断光”进 L1；`-1.8~2.1 dBm` 实测健康区间下沉到 L3。 |
| C6_tx_down_excludes_medium | L1 | “光纤不能解释从未发出的光”是物理排除规则，保留。 |
| C7_rx_power_range | L1 + L3 | “rx 存在连续劣化区间，区别于 tx”进 L1；`-12.3~3.0 dBm` 实测区间下沉到 L3。 |
| C8_tx_ok_rx_down_indicates_medium | L1 | “发出但对端收不到说明光在路径中丢失”是拓扑物理事实；统计富集比例移出。 |
| C9_bidirectional_symmetry | L1 | 双向同 lane 异常指向共享部分、单向异常指向方向端点，保留。 |
| C10_all_lanes_vs_single_lane | L1 | 全 lane 与单 lane 对应共享层级不同，保留。 |
| C11_media_snr_floor | L3 | `22.5 dB` 是训练集实测正常带下界，作为数值树候选分裂点。 |
| C12_no_absolute_link_loss | L2 | 两端功率不能直接相减是本数据量测契约，命中时否决损耗推理。 |
| C13_serdes_snr_unit_unknown | L2 | `serdes_snr` 量纲未知是量测契约，禁止按 dB 解释。 |
| C14_host_snr_mostly_missing | L2 | 缺失不等于正常，是量测完整性契约。 |
| C15_blackout_sentinel_is_not_laser_off | L2 | 全链路 blackout 时哨兵语义翻转，是数据有效性契约；命中后不再用断光 token 推理。 |
| C16_receive_symptom_constrains_far_transmit_chain | L1 + L3 | 接收侧看到对端发来的光进 L1；方向不对称的命中率与 Wilson 下界进 L3。 |
| C17_l2_side_receive_symptom_is_not_discriminative | L3 | 这是分位数口径下的负统计结论，由数值树叶子的低下界表达。 |
| C18_single_lane_scope_does_not_exclude_fiber | L1 | 单 lane 只能排除端口级共享原因，不能排除单纤芯介质问题，保留。 |
| C19_population_prior_is_not_case_evidence | L2 | 类别先验、SOP 叶节点分布不是当前 case 证据，作为推理契约。 |
| C20_fiber_not_identifiable_from_current_telemetry | L2 | 当前遥测无法确认 fiber，是可识别性契约，触发补采而非自动结论。 |
| C21_healthy_band_tx_level_is_not_attribution_evidence | L2 + L3 | “正常带内 tx 高低不能归因、两端 tx 相减不可用”进 L2；相关训练集探针统计进 L3。 |
| C22_receive_lane_imbalance_indicates_far_transmit_array | L1 + L3 | 同侧接收极差消掉共模项的物理解释进 L1；`6/7`、Wilson 下界等统计进 L3。 |
| C23_expert_receive_anomaly_on_l1_supports_l2 | L1 + L3 | 接收侧指向对端进 L1；专家阈值命中分布与下界进 L3。 |
| C24_expert_receive_anomaly_on_l2_supports_l1 | L1 + L3 | C23 镜像的物理方向进 L1；专家阈值口径下的统计可靠性进 L3。 |
| C25_expert_local_chain_anomaly_on_l1_supports_l1 | L1 + L3 | 发送/电口读数度量本端信号进 L1；规则组可靠性进 L3。 |
| C26_expert_local_chain_anomaly_on_l2_is_not_discriminative | L3 | 这是多数类条件下的负统计结论，由数值树叶子的低下界表达。 |

## 迁移后的使用原则

- 高相似度分支只复用历史表象，不注入 L1/L2/L3。
- 中相似度分支只注入 L1 与 L2，用来判断缺失证据是否关键。
- 低相似度分支注入 L1、L2，以及 L3 决策树路径；L3 只能作为统计先验，不能写入物理约束引用。
- 任何新增“物理约束”如果包含训练集命中数、类别比例、Wilson 下界或分位数边界，必须进入 L3，而不是 L1。
