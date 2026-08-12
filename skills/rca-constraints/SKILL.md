---
name: rca-constraints
description: 光链路 RCA 的物理约束库（M5）。在 N5b 补证据与 N5c 通用排障推理时注入，用于约束 LLM 的每一步推断。本文件由 scripts/render_constraint_skill.py 从 rca_framework/constraints/library.py 自动生成，不要手工编辑。
---

# 光模块物理约束库

版本 `constraint-library-v6`，内容指纹 `af09f49aba8039ca`，共 26 条。
`measured` 类参数的实测口径：rca_v2_l2fixed manifest train split（seed=42，train_ratio=0.6）。

## 使用方式

1. N5c（低匹配 / 未见模式）必须注入全部约束。
2. N5b（部分匹配）只在需要补证据或仲裁冲突时注入相关类别。
3. 推理的每一步都要能指到具体的约束 ID；指不到就说明该步没有物理依据。
4. 执行顺序固定：先用排除条件砍掉不可能的根因，再看禁止推断避免走进死胡同，
   最后才用倾向性线索排序剩余候选。

## 约束清单

| ID | 类别 | 类型 | 断言 | 参数来源 | 审核状态 |
| --- | --- | --- | --- | --- | --- |
| `C1_bias_zero_means_laser_off` | bias_current | 物理恒等 | 偏置电流为零等价于该 lane 未发光 | device_spec（断光哨兵=-39 dBm；零电流判定=bias == 0 mA） | pending_expert_review |
| `C2_bias_healthy_band` | bias_current | 倾向性线索 | 健康 lane 的偏置电流落在窄带内 | measured（下界=7.2 mA；上界=7.8 mA） | pending_expert_review |
| `C3_temperature_operating_range` | temperature | 排除条件 | 模块温度在工作范围内则排除热致故障 | device_spec（下界=0 degC；上界=70 degC） | pending_expert_review |
| `C4_voltage_nominal_band` | voltage | 排除条件 | 供电电压在 3.3 V ±5% 内则排除供电异常 | device_spec（标称=3.3 V；允许偏差=±5%） | pending_expert_review |
| `C5_tx_power_range` | tx_power | 物理恒等 | 发送光功率的量程与断光哨兵 | measured（断光哨兵=-39 dBm；健康下界=-1.8 dBm；健康上界=2.1 dBm） | pending_expert_review |
| `C6_tx_down_excludes_medium` | tx_power | 排除条件 | 本端未发光时排除介质根因 | derived（断光哨兵=-39 dBm） | pending_expert_review |
| `C7_rx_power_range` | rx_power | 物理恒等 | 接收光功率的量程与断光哨兵 | measured（断光哨兵=-39 dBm；实测下界=-12.3 dBm；实测上界=3.0 dBm） | pending_expert_review |
| `C8_tx_ok_rx_down_indicates_medium` | lane_directional_consistency | 倾向性线索 | 本端发光正常而对端同 lane 无光指向介质或对端接收 | derived（断光哨兵=-39 dBm） | pending_expert_review |
| `C9_bidirectional_symmetry` | lane_directional_consistency | 倾向性线索 | 双向对称异常指向介质，单向异常指向该方向的端点 | device_spec（无参数） | pending_expert_review |
| `C10_all_lanes_vs_single_lane` | lane_directional_consistency | 倾向性线索 | 全 lane 同时异常与单 lane 异常指向不同层级 | device_spec（无参数） | pending_expert_review |
| `C11_media_snr_floor` | signal_quality | 倾向性线索 | 介质侧信噪比显著低于正常带且收光正常时指向链路质量 | measured（正常带下界=22.5 dB；正常带中位数=25.6 dB (L1) / 26.0 dB (L2)） | pending_expert_review |
| `C12_no_absolute_link_loss` | measurement_validity | 禁止推断 | 本数据集不能用两端功率相减计算链路损耗 | measured（无参数） | pending_expert_review |
| `C13_serdes_snr_unit_unknown` | measurement_validity | 禁止推断 | serdes_snr 不是 dB 量纲，不得按信噪比解释 | measured（健康区间=约 6.6e5 - 8.3e5（量纲未知）；失效哨兵=1） | pending_expert_review |
| `C14_host_snr_mostly_missing` | measurement_validity | 禁止推断 | host_snr 在多数 case 上缺失，缺失不等于正常 | measured（训练集存在率=52/161（32.3%）） | pending_expert_review |
| `C15_blackout_sentinel_is_not_laser_off` | measurement_validity | 禁止推断 | 全链路读数同时触底时，哨兵表示「未读到数」而不是「无光」 | measured（断光哨兵=-39 dBm；训练集命中=4/161） | pending_expert_review |
| `C16_receive_symptom_constrains_far_transmit_chain` | attribution_direction | 倾向性线索 | 接收侧症状把故障约束在对端发送链路、介质与本端接收链路三者之内 | derived（接收侧症状定义=RxLOS / RxLOL 告警，或 rxpower / media_snr 存在断 lane） | pending_expert_review |
| `C17_l2_side_receive_symptom_is_not_discriminative` | attribution_direction | 禁止推断 | L2 侧接收症状不足以支持 L1 根因 | measured（训练集触发=25/161；对端归因正确率=12/25 = 48.0%；Wilson 95% 下界=30.0%；L1 类别先验=30.4%） | pending_expert_review |
| `C18_single_lane_scope_does_not_exclude_fiber` | lane_directional_consistency | 禁止推断 | 单 lane 异常缩小的是共享层，不是介质本身 | measured（训练集单 lane 断 case 数=50（L1 侧 37 + L2 侧 13）） | pending_expert_review |
| `C19_population_prior_is_not_case_evidence` | measurement_validity | 禁止推断 | 类别先验与 SOP 叶节点分布不是本 case 的物理证据 | derived（训练集 L2 先验=100/161 = 62.1%） | pending_expert_review |
| `C20_fiber_not_identifiable_from_current_telemetry` | identifiability | 禁止推断 | 现有遥测无法识别 fiber 根因 | measured（fiber 全局先验=12/161 = 7.45%；最强富集条件=L2 侧 rx 单 lane 断：3/13 = 23.1%，Wilson 下界 8.2%） | pending_expert_review |
| `C21_healthy_band_tx_level_is_not_attribution_evidence` | tx_power | 禁止推断 | 正常带内的发送功率高低不是归因证据，两端相减更不是 | measured（标签 L1 的 case=L1 侧 tx 中位 +0.860 dBm / L2 侧 +0.863 dBm；标签 L2 的 case=L1 侧 tx 中位 +0.835 dBm / L2 侧 +0.855 dBm；两端相减探针与 tx 低尾 token 的 Jaccard=0.65） | pending_expert_review |
| `C22_receive_lane_imbalance_indicates_far_transmit_array` | lane_directional_consistency | 倾向性线索 | 同侧接收 lane 间不均衡指向对端发送阵列的通道差异 | measured（L2 侧接收不均衡命中=7/161；其中根因为对端 L1=6/7 = 85.7%；Wilson 95% 下界=48.7%；L1 类别先验=30.4%） | pending_expert_review |
| `C23_expert_receive_anomaly_on_l1_supports_l2` | attribution_direction | 倾向性线索 | L1 侧接收类读数越出工程阈值时，故障指向对端 L2 的发送链路 | measured（rxpower 判异阈值=断光 -40 dBm / 低值 < -2.5 dBm / 高值 > 4.6 dBm / lane 极差 > 1 dB；media_snr 判异阈值=断光 0 / 低值 < 22.4 / 高值 > 28.7 / lane 极差 > 3；训练集触发=58/161；对端归因正确率=46/58 = 79.3%；Wilson 95% 下界=67.2%；L2 类别先验=62.1%） | pending_expert_review |
| `C24_expert_receive_anomaly_on_l2_supports_l1` | attribution_direction | 倾向性线索 | L2 侧接收类读数越出工程阈值时，故障指向对端 L1 的发送链路 | measured（训练集触发=22/161；对端归因正确率=13/22 = 59.1%；Wilson 95% 下界=38.7%；L1 类别先验=30.4%） | pending_expert_review |
| `C25_expert_local_chain_anomaly_on_l1_supports_l1` | attribution_direction | 倾向性线索 | L1 侧发送与电口读数越出工程阈值时，故障指向 L1 自身 | measured（训练集触发=18/161；本端归因正确率=11/18 = 61.1%；Wilson 95% 下界=38.6%；L1 类别先验=30.4%） | pending_expert_review |
| `C26_expert_local_chain_anomaly_on_l2_is_not_discriminative` | attribution_direction | 禁止推断 | L2 侧发送与电口读数越阈不足以支持 L2 | measured（训练集触发=30/161；本端归因正确率=19/30 = 63.3%；Wilson 95% 下界=45.5%；L2 类别先验=62.1%） | pending_expert_review |

## 逐条说明

### C1_bias_zero_means_laser_off — 偏置电流为零等价于该 lane 未发光

- **物理依据**：激光器的光输出由偏置电流驱动。偏置电流为 0 mA 时器件根本没有被点亮，该 lane 必然没有光输出，这与光纤链路的状态无关。
- **形式表达**：`bias[side][lane] == 0  <=>  txpower[side][lane] <= -39 dBm`
- **实测证据**：训练集 1008 个 lane 读数中有 45 个 bias == 0，涉及 8 条 case；这 45 个 lane 的 txpower 全部同时处于断光哨兵，无一例外，反向也成立：没有出现 tx 断光而 bias 非零的 lane。
- **诊断用法**：先用它把「没发出光」与「光发出后丢失」分开。前者的根因在发送端设备，后者才可能是介质。这一步必须在任何功率比较之前做。

### C2_bias_healthy_band — 健康 lane 的偏置电流落在窄带内

- **物理依据**：同型号模块在正常工作点上的偏置电流分布很窄。电流显著抬高通常意味着激光器老化后需要更大驱动才能维持同样光功率，是器件劣化的早期信号。
- **形式表达**：`7.2 mA <= bias[side][lane] <= 7.8 mA  (healthy lane)`
- **实测证据**：训练集非零偏置电流的 p25 = 7.22 mA（L1）/ 7.29 mA（L2），p99 = 7.72 mA（L1）/ 7.78 mA（L2），全部落在 7.2-7.8 mA。
- **诊断用法**：本数据集内没有观察到落在该带之外的非零电流，因此它当前只能作为「电流正常」的排除依据，不能作为老化判据。合并数据集后重新标定。

### C3_temperature_operating_range — 模块温度在工作范围内则排除热致故障

- **物理依据**：商用光模块的工作温度范围是 0-70 °C。温度在范围内时，温漂不足以导致链路中断或降 lane。
- **形式表达**：`0 degC <= Temperature[side] <= 70 degC`
- **实测证据**：训练集 252 个温度读数全部落在 39.35-60.13 °C，无一超出 70 °C。L2 侧系统性高于 L1 侧约 3.5 °C（中位数 50.30 vs 46.72），这是 200G 与 400G 端口的形态差异，不是故障。
- **诊断用法**：在本数据集上温度恒为排除条件：不允许把根因写成过温。L1 / L2 之间的固定温差也不能当作两侧差异证据。

### C4_voltage_nominal_band — 供电电压在 3.3 V ±5% 内则排除供电异常

- **物理依据**：光模块标称供电 3.3 V，允许偏差 ±5%，即 3.135-3.465 V。超出下界时激光器驱动与 DSP 都可能不稳定。
- **形式表达**：`3.135 V <= Voltage[side] <= 3.465 V`
- **实测证据**：训练集 252 个电压读数中只有 1 个低于下界（case_aa307cc7c7db 的 L2 侧 3.10 V），其余全部落在 3.14-3.42 V。
- **诊断用法**：命中越界时把该侧设备列为候选并要求人工确认；由于全训练集只有 1 例，不允许据此建立统计规则。

### C5_tx_power_range — 发送光功率的量程与断光哨兵

- **物理依据**：单 lane 发送光功率由激光器输出决定，正常工作点集中在 0 dBm 附近的窄带；读数掉到 -39 dBm 及以下是「无光」的哨兵值，不是一个真实的功率测量。
- **形式表达**：`txpower[side][lane] > -39 dBm  =>  -1.8 dBm <= txpower <= 2.1 dBm`
- **实测证据**：训练集健康 txpower 共 963 个读数，L1 侧 -1.70~1.91 dBm，L2 侧 -1.73~2.08 dBm；没有介于 -39 dBm 与 -1.8 dBm 之间的中间值，说明发送功率是「要么正常要么无光」。
- **诊断用法**：发送功率不存在渐变劣化区间，因此 tx 侧只需判断有光 / 无光，不要对 tx 做「偏低多少 dB」的推断。

### C6_tx_down_excludes_medium — 本端未发光时排除介质根因

- **物理依据**：光纤只能衰减已经进入它的光，不能解释一束从未被发出的光。本端某 lane 没有光输出时，该方向的问题必然在发送端。
- **形式表达**：`txpower[near][lane] <= -39 dBm  =>  root_cause != fiber (for that direction)`
- **实测证据**：由 C1 与 C5 推出，不引入新参数。训练集中 tx 断光的 lane 与 bias == 0 完全重合。
- **诊断用法**：这是排除 fiber 的最强单条依据，应在 N5c 推理的第一步执行。

### C7_rx_power_range — 接收光功率的量程与断光哨兵

- **物理依据**：接收功率等于对端发送功率减去链路衰减，因此它有真实的连续劣化区间；读数掉到 -39 dBm 及以下同样是「无光」哨兵。
- **形式表达**：`rxpower[side][lane] > -39 dBm  =>  -12.3 dBm <= rxpower <= 3.0 dBm`
- **实测证据**：训练集健康 rxpower 共 929 个读数，L1 侧 -12.15~2.95 dBm，L2 侧 -12.25~2.83 dBm；与 tx 不同，rx 存在连续的低功率区间（p1 为 -8.45 / -4.68 dBm）。
- **诊断用法**：接收侧允许讨论「偏低多少」，这是与发送侧的关键区别，也是 `level_tail` 特征家族只在接收侧有判别力的物理原因。

### C8_tx_ok_rx_down_indicates_medium — 本端发光正常而对端同 lane 无光指向介质或对端接收

- **物理依据**：光已经被发出却没有到达对端，说明它在传输路径上丢失。路径包含光纤、连接器、以及对端的接收器件。
- **形式表达**：`txpower[near][lane] > -39 dBm AND rxpower[far][lane] <= -39 dBm  =>  root_cause in {fiber, far_end_device}`
- **实测证据**：全量 211 条 case 中有 61 条命中该模式，标签分布 L2 40 / L1 13 / fiber 8。fiber 在命中组中的占比 13.1%，高于全局占比 6.6%，约 2 倍富集；但它同时命中 53 条非 fiber case，因此只能作为 indicator 而不是判据。
- **诊断用法**：命中后必须继续区分「介质」与「对端接收器件」，单靠这一条不能判 fiber。区分手段见 C9。

### C9_bidirectional_symmetry — 双向对称异常指向介质，单向异常指向该方向的端点

- **物理依据**：一对光纤中的两根分别承载两个方向。同一 lane 双向同时异常说明整条 lane 或其光纤对被共同影响（例如同一根尾纤被拔出、同一个连接器脏污）；只有单向异常说明问题落在该方向的发送端或接收端，而不是共享的介质。
- **形式表达**：`abnormal(L1->L2, lane) AND abnormal(L2->L1, lane)  =>  shared cause (fiber pair / connector); XOR  =>  endpoint of that direction`
- **实测证据**：全量 211 条中同 lane 双向异常的有 9 条（标签 L2 6 / fiber 2 / L1 1），样本量太小，不足以支撑统计结论；这条约束的依据是物理拓扑而非数据。
- **诊断用法**：与 C8 串联使用：C8 判断光是否在路径中丢失，C9 判断丢失是否双向对称。双向对称才把 fiber 提到首位。

### C10_all_lanes_vs_single_lane — 全 lane 同时异常与单 lane 异常指向不同层级

- **物理依据**：一个端口的所有 lane 共享供电、时钟、模块壳体和同一束光纤；单条 lane 则有独立的激光器、探测器和纤芯。所有 lane 同时异常指向共享层，单 lane 异常指向该通道的独立器件。
- **形式表达**：`down_lane_count == lane_count  =>  port-level; down_lane_count == 1  =>  channel-level`
- **实测证据**：这正是特征家族 `signal_drop` 把断 lane 数分成 single_lane / partial_lanes / all_lanes 三档的物理依据；T1 家族消融显示该分档是 v1 中不可替代的一项。
- **诊断用法**：决定排障动作的粒度：整端口换模块 / 换整束纤，还是单通道定位。

### C11_media_snr_floor — 介质侧信噪比显著低于正常带且收光正常时指向链路质量

- **物理依据**：介质侧 SNR 反映解调后的信号质量。收光功率正常而 SNR 偏低，说明损伤不是功率衰减，而是色散、反射、串扰这类不改变总功率的链路质量问题。
- **形式表达**：`rxpower normal AND media_snr < 22.5 dB  =>  link quality degradation`
- **实测证据**：训练集健康 media_snr 的 p1 为 22.47 dB（L1）/ 22.95 dB（L2），低于 20 dB 的读数只有 4 个（16.71 / 17.51 / 17.51 / 17.70）。
- **诊断用法**：触发极少，当前只能作为个别 case 的补充线索；不要把它写进任何需要统计支撑的规则。

### C12_no_absolute_link_loss — 本数据集不能用两端功率相减计算链路损耗

- **物理依据**：无源链路的损耗必然非负，即对端收到的功率不可能高于本端发出的功率。本数据集违反这一点，说明两端 lane 编号不对应，或收发功率的标定口径不同。
- **形式表达**：`mean(txpower[near]) - mean(rxpower[far]) >= 0   # 本数据集不成立`
- **实测证据**：按 lane 号配对后，L1->L2 方向的均值损耗中位数为 -0.285 dB，L2->L1 为 -0.227 dB，两个方向的中位数都是负值，物理上不可能。legacy `directional_loss` 学到的上界（3.11 / 3.42 dB）因此也不可信。第二个更直观的证据（迭代 1 补测）：只看序不看数值，统计两端**最差 lane 的编号**是否相同。若两端编号真的对应，故障 case 里这个比例应明显高于随机；实测 rxpower 为 37/155 = 23.9%、media_snr 为 46/161 = 28.6%，而 4 lane 下随机一致的概率就是 25%，两者都与随机无法区分。这个检验只需要遥测本身，可以直接用来判断一份数据的两端 lane 是否对齐。
- **诊断用法**：禁止在约束、规则或 prompt 中写绝对损耗门限。只允许使用同侧内部的相对量（lane 间极差）与训练集分位分档。

### C13_serdes_snr_unit_unknown — serdes_snr 不是 dB 量纲，不得按信噪比解释

- **物理依据**：`serdes_snr` 字段的健康取值在 6.6e5-8.3e5 量级，断链时为 1。这不是任何 dB 口径的信噪比，更接近某种原始计数或定点数。
- **形式表达**：`serdes_snr 量纲未知；仅可用作有效 / 无效的二值判断`
- **实测证据**：训练集 972 个 serdes_snr 读数中，健康值 p25-p99 为 6.6e5-8.2e5，最小值为 1。legacy 规则里出现频率很高的 `low_outlier:*:serdes_snr` 就建立在这个量纲未知的字段上。
- **诊断用法**：在向厂商确认量纲之前，只允许用它区分「有效」与「失效」，不允许出现「serdes SNR 低了 x dB」这类表述。

### C14_host_snr_mostly_missing — host_snr 在多数 case 上缺失，缺失不等于正常

- **物理依据**：主机侧信噪比反映模块与交换芯片之间的电口质量，与光链路无关。该字段在本数据集大面积缺失。
- **形式表达**：`host_snr 存在率 = 52/161 训练 case`
- **实测证据**：rca_v2_l2fixed manifest train split 的 161 条 case 中，只有 52 条任一侧有非空 host_snr 读数，109 条两侧均无有效读数（存在率 32.3%）。
- **诊断用法**：缺失必须被显式表述为「未采集」，不能当作「正常」。N6 在判断证据充分性时要把它算作缺失证据而不是通过项。

### C15_blackout_sentinel_is_not_laser_off — 全链路读数同时触底时，哨兵表示「未读到数」而不是「无光」

- **物理依据**：两端的发送、接收、介质侧信噪比同时全部处于断光哨兵，同时 TxLOS 却报 Normal，这两件事互相矛盾：模块若真的没有发光，TxLOS 应当告警。更合理的解释是链路整体中断后遥测通道本身失效，所有光学读数回落到哨兵默认值。此时哨兵是「读不到」而不是「没有光」。
- **形式表达**：`ALL(txpower, rxpower over both sides) <= -39 dBm AND TxLOS == Normal  =>  哨兵含义为 no_reading，不得据此推断激光关断`
- **实测证据**：rca_v2_l2fixed manifest train split 的 161 条中有 4 条命中：两侧 4 个 lane 的 txpower / rxpower 全为 -40.0 dBm，状态位一律 TxLOS=Normal、TxLOL=Normal、RxLOS=Abnormal、RxLOL=Abnormal。4 条的标签为 L2 3 条、fiber 1 条：物理观测完全一致而根因不同，说明该状态下的遥测不足以区分根因。
- **诊断用法**：这是 C6 的前置条件：只有在遥测确实有效（存在任一非哨兵读数）时，才允许用「本端未发光」去排除介质根因。命中本约束的 case 应直接转人工，不论它产出了多少个特征 token——token 多不等于证据强。

### C16_receive_symptom_constrains_far_transmit_chain — 接收侧症状把故障约束在对端发送链路、介质与本端接收链路三者之内

- **物理依据**：一侧的接收类读数（rxpower、media_snr、RxLOS / RxLOL）度量的是**对端发出、穿过介质之后到达本端**的光。因此接收侧出现症状时，候选根因只能落在「对端发送链路」「介质」「本端接收链路」这三段里，在物理上不可能是本端自己的发送链路——本端发出的光根本不经过本端的接收器。这条方向性是光链路 RCA 里最基本的归因约束：报症状的一端通常不是肇事的一端。
- **形式表达**：`receive_symptom(X)  =>  root_cause_chain in {tx_chain(Y), medium, rx_chain(X)}  AND  root_cause_chain != tx_chain(X)`
- **实测证据**：rca_v2_l2fixed manifest train split（161 条）按「只有哪一侧出现接收侧症状」分组：只有 L1 侧 63 条，其中根因为对端 L2 的 43 条 = 68.3%（Wilson 下界 56.0%，L2 先验 62.1%）；只有 L2 侧 25 条，其中根因为对端 L1 的 12 条 = 48.0%（Wilson 下界 30.0%，L1 先验 30.4%）。**两个方向不对称：只有 L1 受害方向的下界超过其预测类别的先验。**更细的口径同样如此：L1 侧 rx 只有单 lane 断（其余 lane 健康）37 条，根因 L2 30 条 = 81.1%（下界 65.8%）；镜像条件 L2 侧 rx 单 lane 断只有 13 条，根因 L1 仅 4 条 = 30.8%，与 L1 先验无法区分。
- **诊断用法**：只允许在 L1 侧为接收受害方时用它支持 L2；L2 侧为受害方时按 C17 处理，不得镜像套用。这条不对称本身是实测结果，不要为了对称性把它写成双向规则。

### C17_l2_side_receive_symptom_is_not_discriminative — L2 侧接收症状不足以支持 L1 根因

- **物理依据**：C16 的方向性在物理上是对称的，但在本数据集上只有一个方向具备统计判别力。L2（200G）侧作为接收受害方时，对端归因的实测正确率与 L1 的类别先验没有区别，说明现有遥测无法把「L1 发送链路劣化」与「L2 自身接收链路劣化」分开——两者在 L2 侧看到的现象一样。
- **形式表达**：`receive_symptom(L2) AND NOT receive_symptom(L1)  =>  P(L1) 与先验不可区分；不得据此断言 L1`
- **实测证据**：rca_v2_l2fixed manifest train split：只有 L2 侧出现接收症状的 25 条中，标签为 L1 的 12 条、L2 的 10 条、fiber 的 3 条。预测 L1 的 Wilson 下界 30.0% 恰好落在 L1 先验 30.4% 上，没有增益。单 lane 口径更差：L2 侧 rx 单 lane 断 13 条中 L1 仅 4 条（30.8%）。
- **诊断用法**：命中本约束的 case 应输出「候选 L1，但当前证据不足以定论」并给出补采项（L1 侧 host_snr / serdes 读数、L1 侧同 lane 的发送功率历史），而不是给出 L1 结论。这是把一个实测负结果变成明确的补采动作。唯一的例外是 C22：L2 侧各 lane 之间接收不齐（而不是整体偏低）仍可作为 L1 的弱支持，两者的适用 token 不重叠。

### C18_single_lane_scope_does_not_exclude_fiber — 单 lane 异常缩小的是共享层，不是介质本身

- **物理依据**：单条 lane 异常而同端口其余 lane 健康，可以排除所有 lane 共享的部分：模块供电、壳体温度、整束光纤被拔出、整个连接器脱落。但**不能排除介质**：并行光模块的每条 lane 走独立纤芯，单根纤芯断裂或单个 MPO 芯位脏污同样只影响一条 lane。把「单 lane」直接推成「不是光纤」是一个很自然但错误的推理。
- **形式表达**：`down_lane_count == 1  =>  排除 port 级共享原因；不得推出 root_cause != fiber`
- **实测证据**：rca_v2_l2fixed manifest train split 中 rx 恰好一条 lane 断的 case 共 50 条，其中 fiber 标签 6 条 = 12.0%，**高于** fiber 全局先验 7.45%。如果单 lane 能排除介质，这个比例应当低于先验。
- **诊断用法**：单 lane 结论只能写成「排除端口级共享原因」，后续仍要在「对端该通道的激光器」「该 lane 的纤芯 / 芯位」「本端该通道的探测器」之间区分。

### C19_population_prior_is_not_case_evidence — 类别先验与 SOP 叶节点分布不是本 case 的物理证据

- **物理依据**：learned SOP 的叶节点标签分布、历史候选的标签投票和类别先验都是**群体统计**。它们可以决定在没有判别证据时的默认动作，但它们不描述当前这条链路发生了什么。把它们当成证据会产生一种特别难发现的错误：结论看起来有依据，实际上整条推理链没有引用任何一个当前 case 的观测。
- **形式表达**：`SOP_leaf_distribution, class_prior, history_label_vote  NOT IN cited_evidence  AND  effect != support`
- **实测证据**：MVP 正式实验中 M9 前的 44 个候选有 23 个正确（52.3%），而单纯预测多数类 L2 在同一测试集上是 62.6%。也就是说，一条大量引用群体先验的推理链的表现低于直接报多数类，它增加的只是解释的外观。
- **诊断用法**：M7 应拒绝把 SOP 路径、叶节点分布或历史标签投票写进 `cited_evidence` 的回答；允许在自然语言里提到它是默认动作的来源，但不允许作为 support 步骤。

### C20_fiber_not_identifiable_from_current_telemetry — 现有遥测无法识别 fiber 根因

- **物理依据**：介质根因需要的证据是链路损耗、反射事件位置、端面污染或弯曲损耗，这些都要靠 OTDR、端面镜检或双向功率标定获得。本数据集的遥测只有两端模块的自报读数，且按 C12 连绝对损耗都算不出来。因此 fiber 在信息层面就不可识别，这不是模型能力问题。
- **形式表达**：`max over observable conditions of P(fiber | condition) 的 Wilson 95% 下界 <= 0.082  =>  不得断言 fiber`
- **实测证据**：在 rca_v2_l2fixed manifest train split 上穷举了断 lane 波及范围、双向同 lane 断、两侧 media_snr 同时偏低、两侧收光同时偏弱等条件，支持数 >= 6 的条件里 fiber 占比最高为 23.1%（n=13），Wilson 95% 下界 8.2%，与 7.45% 的先验无法区分。MVP 正式实验的测试侧也一致：系统预测 fiber 10 次只对 1 次。
- **诊断用法**：禁止输出 fiber 结论。命中疑似介质模式时输出「候选 fiber，需现场确认」并请求 OTDR 曲线、端面镜检或双向功率标定，让 fiber 成为一个明确的补采分支而不是一个低精度的猜测。

### C21_healthy_band_tx_level_is_not_attribution_evidence — 正常带内的发送功率高低不是归因证据，两端相减更不是

- **物理依据**：按 C5，发送功率只有「正常」与「无光」两态，正常带内的高低由激光器个体差异、出厂标定和端口形态决定，不由链路故障决定。在这样一个与根因几乎无关的连续量上取分布尾部，仍然会得到看起来偏斜的标签分布，因为尾部样本少。这类关联是抽样波动，不是物理关系。把两端的发送功率相减会同时踩上 C12 的坑：两端标定口径本来就不可比。
- **形式表达**：`txpower[side] > -39 dBm  =>  txpower 的具体数值不得进入 support 步骤；禁止使用 mean(txpower[L1]) - mean(txpower[L2])`
- **实测证据**：rca_v2_l2fixed manifest train split（161 条）三项实测：（1）按标签分层的健康 tx 均值中位数几乎相同（见 parameters），即发送电平与根因基本无关；（2）`level:L1:txpower_mean:low_tail` 命中 39 条且**无一条含断光哨兵**，标签 L2 29 / L1 6 / fiber 4，precision 74.4% 但 Wilson 下界 58.9%，低于 L2 先验 62.1%，因此没有增益；（3）两端相减的探针 `probe:txpower_side_gap:L1_worse` 是唯一下界（65.8%）超过 L2 先验的 tx 类信号，但它与上面那个低尾 token 的 Jaccard 达 0.65，控制该 token 后剩余支持只有 7 条，增益消失。
- **诊断用法**：发送侧一律只做有光 / 无光判断（C5、C6）。不允许出现「L1 侧发送功率偏低所以……」这类步骤，也不允许两端功率相减。这条约束的作用是拦掉一条统计上很诱人、物理上站不住的捷径。

### C22_receive_lane_imbalance_indicates_far_transmit_array — 同侧接收 lane 间不均衡指向对端发送阵列的通道差异

- **物理依据**：并行光模块的每条 lane 有独立的激光器与探测器，但同一端口内所有 lane 共享标定口径、整束光纤的共模损耗和接收侧的 AGC 配置。因此同侧各 lane 接收功率之间的**极差**天然消掉了这些共模项，剩下的差异只能来自对端各发送通道之间的不一致，即对端发送阵列的通道级劣化。这使它比两端绝对电平相减可靠得多——后者按 C12 在本数据集上根本不成立。「用同侧相对量做跨端归因」是这份数据里唯一站得住的跨端推理方式。
- **形式表达**：`spread(rxpower[X]) 显著大于同侧正常波动  =>  support tx_array(Y)，Y 为对端`
- **实测证据**：rca_v2_l2fixed manifest train split：`imbalance:L2:rxpower` 命中 7 条，标签为 L1 的 6 条、fiber 1 条，无一条含断光哨兵（即不均衡不是断 lane 造成的）。Wilson 下界 48.7% 超过 L1 先验 30.4%，这是全训练集上**唯一**一个下界超过 L1 先验的观测条件。镜像方向 `imbalance:L1:rxpower` 命中 10 条、8 条为 L2（80.0%，下界 49.0%），但 L2 先验是 62.1%，所以镜像方向不成立。这个不对称主要来自两类先验相差一倍（30.4% vs 62.1%）——支持少数类需要的证据强度本来就更低，不要读成物理上的不对称。
- **诊断用法**：只允许用它支持 L1，且必须标注为弱证据：支持数只有 7 条，下界 48.7% 意味着「比先验强，但远达不到可以定论」。命中后应输出 L1 候选并请人工确认，同时建议补采 L1 侧各发送通道的功率与偏置电流历史，用来确认是哪一路通道。本条是 C17 的细化，不是推翻：C17 否掉的是「L2 侧整体收光低或告警」，本条针对的是「L2 侧各 lane 之间不齐」，两者不可混用。

### C23_expert_receive_anomaly_on_l1_supports_l2 — L1 侧接收类读数越出工程阈值时，故障指向对端 L2 的发送链路

- **物理依据**：接收侧读数度量的是**对端发出、穿过介质之后**到达本端的光，因此接收类指标（rxpower、media_snr）越出工程阈值时，故障不可能出在本端自己的发送器上。现网专家规则据此把「rxpower 异常」「media_snr 异常」以及三项组合异常（serdes_snr + media_snr + rxpower 同侧同时异常）统一定界到**异常所在端的对端**。组合异常的物理含义更强：光口、介质侧信噪比与电口三级同时劣化，说明整条接收通道拿到的信号本身就是坏的，而不是本端某一级的问题。
- **形式表达**：`expert_receive_anomaly(L1)  =>  support L2  其中 expert_receive_anomaly 按 EXPERT_EXPERIENCE.md §3.3 的固定阈值判定`
- **实测证据**：rca_v2_l2fixed manifest train split：按专家阈值判定后由 L1 侧接收类异常（含三项组合异常）胜出仲裁的 58 条中，根因确为对端 L2 的 46 条 = 79.3%，Wilson 下界 67.2% 超过 L2 先验 62.1%。留出集同向：34 条中 28 条 = 82.4%（下界 66.5%，先验 62.6%）。两个切分的下界都超过先验，且规则不含任何在本数据集上拟合的参数。
- **诊断用法**：命中即可支持 L2。它与 C16 说的是同一段物理关系，但证据口径不同：C16 用的是分位数 token（`drop:L1:rxpower:` 等），本条用的是工程阈值 token（`expert:L1:...`）。同一条 case 可能只命中其中一种，两者可以互相印证，但不得把它们当成两条独立证据——它们读的是同一批原始读数。

### C24_expert_receive_anomaly_on_l2_supports_l1 — L2 侧接收类读数越出工程阈值时，故障指向对端 L1 的发送链路

- **物理依据**：C23 的镜像方向。物理上这条关系本来就是对称的——接收侧看到的永远是对端发出的光——此前之所以只承认 L1 受害方向（C16 / C17），是因为在分位数口径下 L2 方向测不出增益。
- **形式表达**：`expert_receive_anomaly(L2)  =>  support L1`
- **实测证据**：rca_v2_l2fixed manifest train split：由 L2 侧接收类专家异常胜出仲裁的 22 条中，根因确为对端 L1 的 13 条 = 59.1%，Wilson 下界 38.7% 超过 L1 先验 30.4%。留出集 27 条中 17 条 = 63.0%（下界 44.2%）。**与 C17 的关系必须写清楚**：C17 在分位数口径（`drop:L2:rxpower:` 等）下测到下界 30.0%、与先验 30.4% 无法区分，因此判定该方向不可用。换成工程阈值口径后同一段物理关系变得可用，说明 C17 否掉的是那一种证据定义，不是这段物理关系。这也是迭代 3 最重要的方法论结论：**「现有遥测无法判别 X」这类负结论必须连同异常定义一起陈述，否则会被过度推广。**
- **诊断用法**：命中即可支持 L1，但强度弱于 C23（下界 38.7% vs 67.2%）：支持少数类需要的证据强度本来就低，不要把两者的下界直接比大小。命中时应同时给出补采建议（L1 侧各发送通道的功率与偏置电流历史）。不得与 C17 同时引用：两者读的是同一批原始读数的不同判定口径。

### C25_expert_local_chain_anomaly_on_l1_supports_l1 — L1 侧发送与电口读数越出工程阈值时，故障指向 L1 自身

- **物理依据**：与接收类相反，发送光功率（txpower）、主机侧信噪比（host_snr）与 SerDes 信噪比（serdes_snr）度量的都是**本端自己产生的信号**，光没有出过本端的模块，因此它们越阈时指向本端。这正是专家方向表分成两类的物理依据：看到的是别人发来的光就归对端，看到的是自己发出的信号就归自己。发送侧断光（txpower lane_down）是全表最高优先级——它是一个确定性事实，不需要与其它证据比较强弱。
- **形式表达**：`expert_local_chain_anomaly(L1)  =>  support L1`
- **实测证据**：rca_v2_l2fixed manifest train split：由 L1 侧发送 / 电口类专家异常胜出仲裁的 18 条中，根因确为 L1 的 11 条 = 61.1%，Wilson 下界 38.6% 超过 L1 先验 30.4%。留出集 11 条中 6 条 = 54.5%（下界 28.0%），略低于先验，样本量也小；因此本条只能作为弱支持，不能单独定论。
- **诊断用法**：命中可支持 L1，但必须标注为弱证据并建议人工确认。唯一的例外是 `expert:L1:txpower:lane_down`：本端发送断光是确定性事实，按 C6 还可同时排除 fiber。

### C26_expert_local_chain_anomaly_on_l2_is_not_discriminative — L2 侧发送与电口读数越阈不足以支持 L2

- **物理依据**：C25 的镜像方向在物理上同样成立，但在本数据集上不具备判别力——不是因为物理不对，而是因为 L2 的类别先验已经有 62.1%，一条把 63.3% 的 case 判对的规则并没有比「什么都不看直接报 L2」更好。这是所有面向多数类的证据都要过的一关：**支持多数类需要的下界远高于支持少数类**。
- **形式表达**：`expert_local_chain_anomaly(L2)  =>  P(L2) 与先验不可区分；不得据此断言 L2`
- **实测证据**：rca_v2_l2fixed manifest train split：由 L2 侧发送 / 电口类专家异常胜出仲裁的 30 条中，根因为 L2 的 19 条 = 63.3%，Wilson 下界 45.5% **低于** L2 先验 62.1%。留出集上这一组反而很准（22 条中 20 条 = 90.9%，下界 72.2%），但两个切分差 27 个百分点、训练集下界不达标，按本仓库既定口径不足以升级为 support。记录这个分歧本身有价值：它说明该规则组的可靠性不稳定，需要更多数据才能定论。
- **诊断用法**：命中时只能作为中性观察写进推理链，不得作为 support。若同一条 case 上没有任何其它方向证据，应输出「候选 L2，证据不足以定论」并请求补采 L2 侧发送通道的历史读数。

## 注入 prompt 的文本块

以下内容由 `render_prompt_block()` 产出，是真正进入 prompt 的原文。

```text
# 光模块物理约束（constraint-library-v6，hash af09f49aba8039ca）


## 排除条件：命中后可以直接排除对应根因

- [C3_temperature_operating_range] 模块温度在 0-70 °C 内属于正常工作范围，不构成故障原因。本数据集所有 case 的温度都在此范围内，因此不要把根因归为过温。L2 侧温度普遍比 L1 侧高约 3.5 °C 是端口形态差异，不是异常。（待专家审核）
  结构化引用契约：可用 token 前缀=无需绑定当前 token（仅作中性上下文）；effect 只能为 neutral；target 只能为 空字符串。
- [C4_voltage_nominal_band] 供电电压应在 3.135-3.465 V。落在此范围内则排除供电导致的故障；低于下界时该侧设备是候选根因，但需要人工确认。（待专家审核）
  结构化引用契约：可用 token 前缀=无需绑定当前 token（仅作中性上下文）；effect 只能为 neutral；target 只能为 空字符串。
- [C6_tx_down_excludes_medium] 如果某一侧的 lane 根本没有发出光（txpower 处于 -39 dBm 哨兵），那么该方向的故障不可能由光纤引起，应归到该发送端设备。（待专家审核）
  结构化引用契约：可用 token 前缀=drop:L1:txpower:、drop:L2:txpower:、lane:L1_to_L2:tx_down、lane:L2_to_L1:tx_down；effect 只能为 exclude；target 只能为 fiber。

## 禁止推断：以下推断在本数据集上不成立

- [C12_no_absolute_link_loss] 本数据集两端的光功率读数不能直接相减求链路损耗：实测结果会出现负损耗，说明两端 lane 编号不对应或标定口径不同。不要根据「损耗多少 dB」下结论，只使用同侧 lane 之间的相对差异和有光/无光判断。（待专家审核）
  结构化引用契约：可用 token 前缀=lane:；effect 只能为 neutral；target 只能为 空字符串。
- [C13_serdes_snr_unit_unknown] serdes_snr 字段的量纲未确认，健康值在 6.6e5-8.3e5 量级，失效时为 1。只能把它当作有效 / 失效的二值信号，不要按 dB 信噪比解释或比较。（待专家审核）
  结构化引用契约：可用 token 前缀=serdes:；effect 只能为 neutral；target 只能为 空字符串。
- [C14_host_snr_mostly_missing] 主机侧信噪比 host_snr 在多数 case 中没有采集。看不到该字段时应说明「未采集」，不要推断它正常，也不要用它支持任何结论。（待专家审核）
  结构化引用契约：可用 token 前缀=telemetry:partial_telemetry、telemetry:no_telemetry；effect 只能为 neutral；target 只能为 空字符串。
- [C15_blackout_sentinel_is_not_laser_off] 如果两端的发送与接收光功率全部处于 -39 dBm 哨兵，而 TxLOS 仍报 Normal，说明这是遥测整体失效而不是激光关断。此时不要断言任何一端「没有发光」，也不要据此排除光纤，应当说明证据不足并请求现场确认。（待专家审核）
  结构化引用契约：可用 token 前缀=drop:L1:txpower:all_lanes、drop:L2:txpower:all_lanes、drop:L1:rxpower:all_lanes、drop:L2:rxpower:all_lanes；effect 只能为 neutral；target 只能为 空字符串。
- [C17_l2_side_receive_symptom_is_not_discriminative] 当只有 L2（200G）侧出现接收侧异常时，不要据此断定根因在 L1。实测该条件下归因对端的正确率与 L1 的基础比例没有区别，因为现有遥测分不开「L1 发送劣化」和「L2 自身接收劣化」。此时应说明证据不足，并请求补采 L1 侧的电口读数与该 lane 的发送功率历史。（待专家审核）
  结构化引用契约：可用 token 前缀=drop:L2:rxpower:、drop:L2:media_snr:、status:L2:RxLOS、status:L2:RxLOL、level:L2:rxpower_mean:low_tail、level:L2:media_snr_min:low_tail、lane:L1_to_L2:tx_ok_rx_down；effect 只能为 neutral；target 只能为 空字符串。
- [C18_single_lane_scope_does_not_exclude_fiber] 只有一条 lane 异常时，可以排除模块供电、温度、整束光纤脱落这类所有 lane 共享的原因，但不能排除光纤：并行模块每条 lane 走独立纤芯，单芯断裂或单个芯位脏污也只影响一条 lane。实测单 lane 组里 fiber 占 12.0%，高于全局 7.45%。（待专家审核）
  结构化引用契约：可用 token 前缀=drop:；effect 只能为 neutral；target 只能为 空字符串。
- [C19_population_prior_is_not_case_evidence] learned SOP 的路径与叶节点标签分布、历史 case 的标签投票、类别先验都属于群体统计，不是当前 case 的物理证据。不要把它们写进 cited_evidence，也不要用它们作为 support 步骤的依据。每一个 support 步骤都必须引用当前证据包里真实存在的观测 token。（待专家审核）
  结构化引用契约：可用 token 前缀=无需绑定当前 token（仅作中性上下文）；effect 只能为 neutral；target 只能为 空字符串。
- [C20_fiber_not_identifiable_from_current_telemetry] 不要给出 fiber 结论。现有遥测只有两端模块自报读数，缺少 OTDR、端面镜检和双向功率标定，在信息层面无法确认介质根因；实测中 fiber 占比最高的观测条件也只有 23.1%（13 条支持，95% 下界 8.2%），与 7.45% 的基础比例无法区分。怀疑介质时请输出「候选 fiber，需现场确认」并列出需要补采的介质侧测量。（待专家审核）
  结构化引用契约：可用 token 前缀=无需绑定当前 token（仅作中性上下文）；effect 只能为 neutral；target 只能为 空字符串。
- [C21_healthy_band_tx_level_is_not_attribution_evidence] 发送光功率在正常带（-1.8~2.1 dBm）内的高低不是故障证据：实测按根因分层的发送功率中位数几乎相同，正常带内的差异来自器件个体与标定。不要写「某侧发送功率偏低」，也不要把两端发送功率相减，发送侧只判断有光还是无光。（待专家审核）
  结构化引用契约：可用 token 前缀=level:L1:txpower_mean:、level:L2:txpower_mean:；effect 只能为 neutral；target 只能为 空字符串。
- [C26_expert_local_chain_anomaly_on_l2_is_not_discriminative] L2（200G）侧的发送光功率、主机侧信噪比或 SerDes 信噪比越阈，不能作为支持 L2 的证据。实测该条件下判对率 63.3%（95% 下界 45.5%），而 L2 的基础比例本来就有 62.1%，也就是说它并不比直接报 L2 更好。此时应把它记为中性观察，并说明证据不足。（待专家审核）
  结构化引用契约：可用 token 前缀=expert:L2:txpower:、expert:L2:host_snr:、expert:L2:serdes_snr:、expert:pattern:L2:port_down、expert:points_to:L2:L2；effect 只能为 neutral；target 只能为 空字符串。

## 物理恒等关系

- [C1_bias_zero_means_laser_off] 偏置电流为 0 mA 的 lane 表示激光器未点亮，该 lane 没有光输出。此时该方向的问题在发送端设备，不能归因于光纤。（待专家审核）
  结构化引用契约：可用 token 前缀=drop:L1:bias:、drop:L2:bias:；effect 只能为 neutral；target 只能为 空字符串。
- [C5_tx_power_range] 发送光功率正常时在 -1.8~2.1 dBm，异常时直接掉到 -39 dBm 哨兵值，两者之间没有中间态。因此发送侧只区分有光与无光，不要讨论发送功率轻微偏低。（待专家审核）
  结构化引用契约：可用 token 前缀=drop:L1:txpower:、drop:L2:txpower:、level:L1:txpower_mean:、level:L2:txpower_mean:；effect 只能为 neutral；target 只能为 空字符串。
- [C7_rx_power_range] 接收光功率有真实的连续劣化区间（本数据集健康值 -12.3~3.0 dBm），低于 -39 dBm 表示完全无光。接收侧可以讨论功率偏低的程度，发送侧不可以。（待专家审核）
  结构化引用契约：可用 token 前缀=drop:L1:rxpower:、drop:L2:rxpower:、imbalance:L1:rxpower、imbalance:L2:rxpower、level:L1:rxpower_mean:、level:L2:rxpower_mean:；effect 只能为 neutral；target 只能为 空字符串。

## 倾向性线索：提高可能性，但不足以定论

- [C10_all_lanes_vs_single_lane] 端口内所有 lane 同时异常，指向端口级共享部分：模块本体、供电、或整束光纤。只有一条 lane 异常，指向该通道独立的激光器、探测器或单根纤芯。（待专家审核）
  结构化引用契约：可用 token 前缀=drop:；effect 只能为 neutral；target 只能为 空字符串。
- [C11_media_snr_floor] 介质侧信噪比正常范围约 22.5-27 dB。如果收光功率正常但信噪比明显低于该范围，问题偏向链路质量（色散、反射、连接器端面），而不是功率衰减。（待专家审核）
  结构化引用契约：可用 token 前缀=level:L1:media_snr_min:low_tail、level:L2:media_snr_min:low_tail；effect 只能为 support；target 只能为 fiber。
- [C16_receive_symptom_constrains_far_transmit_chain] 接收侧读数描述的是对端发出、穿过光纤后到达本端的光，因此接收侧异常不可能由本端自己的发送器造成，候选只能是对端发送链路、光纤介质或本端接收链路。在本数据集上，当 L1（400G）侧是接收受害方时，根因落在对端 L2 的实测比例为 81.1%（L1 侧 rx 单 lane 断，37 条支持），可以据此支持 L2。反方向不成立，见 C17。（待专家审核）
  结构化引用契约：可用 token 前缀=drop:L1:rxpower:、drop:L1:media_snr:、status:L1:RxLOS、status:L1:RxLOL、level:L1:rxpower_mean:low_tail、level:L1:media_snr_min:low_tail、lane:L2_to_L1:tx_ok_rx_down；effect 只能为 support；target 只能为 L2。
- [C22_receive_lane_imbalance_indicates_far_transmit_array] 同一侧各 lane 之间的接收功率不齐（极差偏大），指向对端发送阵列中某几路通道的差异，因为同侧 lane 共享标定与整束光纤的共模损耗，极差把这些共模项消掉了。实测中 L2 侧接收不均衡的 7 条里有 6 条根因在对端 L1（95% 下界 48.7%，L1 基础比例 30.4%），可以据此支持 L1，但只能作为弱证据并请人工确认；反方向（L1 侧不均衡支持 L2）不成立。（待专家审核）
  结构化引用契约：可用 token 前缀=imbalance:L2:rxpower；effect 只能为 support；target 只能为 L1。
- [C23_expert_receive_anomaly_on_l1_supports_l2] L1（400G）侧的接收光功率或介质侧信噪比越出工程阈值时，支持根因在对端 L2。理由是接收侧看到的光是对端发出的，本端自己的发送器不在这条光路上。实测 58 条命中里 46 条根因确实在 L2（79.3%，95% 下界 67.2%，L2 基础比例 62.1%）。若 serdes_snr、media_snr、rxpower 三项在同一侧同时异常，这条证据更强。（待专家审核）
  结构化引用契约：可用 token 前缀=expert:L1:rxpower:、expert:L1:media_snr:、expert:pattern:L1:multi_metric、expert:points_to:L1:L2；effect 只能为 support；target 只能为 L2。
- [C24_expert_receive_anomaly_on_l2_supports_l1] L2（200G）侧的接收光功率或介质侧信噪比越出工程阈值时，支持根因在对端 L1。实测 22 条命中里 13 条根因确实在 L1（59.1%，95% 下界 38.7%，L1 基础比例 30.4%）。注意这条与 C17 的适用条件不同：C17 针对的是「相对本数据集分位数偏低」，本条针对的是「越出工程阈值」，后者才具备判别力。同一条 case 不要两条都引。（待专家审核）
  结构化引用契约：可用 token 前缀=expert:L2:rxpower:、expert:L2:media_snr:、expert:pattern:L2:multi_metric、expert:points_to:L2:L1；effect 只能为 support；target 只能为 L1。
- [C25_expert_local_chain_anomaly_on_l1_supports_l1] L1（400G）侧的发送光功率、主机侧信噪比或 SerDes 信噪比越出工程阈值时，支持根因在 L1 自身，因为这些读数度量的是本端自己产生的信号，不经过光纤也不来自对端。实测 18 条命中里 11 条根因在 L1（61.1%，95% 下界 38.6%，L1 基础比例 30.4%），属于弱证据，需人工确认。本端发送断光是例外，它是确定性事实。（待专家审核）
  结构化引用契约：可用 token 前缀=expert:L1:txpower:、expert:L1:host_snr:、expert:L1:serdes_snr:、expert:pattern:L1:port_down、expert:points_to:L1:L1；effect 只能为 support；target 只能为 L1。
- [C2_bias_healthy_band] 健康 lane 的偏置电流在 7.2-7.8 mA。电流在此范围内说明激光器驱动正常，不要把根因归到发送端器件老化。（待专家审核）
  结构化引用契约：可用 token 前缀=drop:L1:bias:、drop:L2:bias:；effect 只能为 neutral；target 只能为 空字符串。
- [C8_tx_ok_rx_down_indicates_medium] 如果本端某 lane 发光正常而对端同一 lane 完全收不到光，说明光在路径中丢失，候选根因是光纤介质或对端接收器件。这条线索会提高光纤的可能性，但不足以定论，必须再结合双向一致性判断。（待专家审核）
  结构化引用契约：可用 token 前缀=lane:L1_to_L2:tx_ok_rx_down、lane:L2_to_L1:tx_ok_rx_down；effect 只能为 support；target 只能为 fiber。
- [C9_bidirectional_symmetry] 同一条 lane 在两个方向上同时异常，说明问题出在双向共享的部分，即光纤对或连接器，光纤是首位候选。如果只有一个方向异常，则问题在该方向的端点设备，光纤应当降位。（待专家审核）
  结构化引用契约：可用 token 前缀=lane:L1_to_L2:bidirectional_same_lane、lane:L2_to_L1:bidirectional_same_lane；effect 只能为 support；target 只能为 fiber。
```

## 待办

- 全部 26 条均为 `pending_expert_review`，需夏思博逐条确认后改为 `approved`。
- `measured` 类参数绑定当前数据集切分，合并数据集到位后必须重测。
- `C12` 与 `C13` 是数据质量问题，需要向厂商确认 lane 编号对应关系与 `serdes_snr` 量纲。
