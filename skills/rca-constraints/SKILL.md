---
name: rca-constraints
description: 光链路 RCA 的物理约束库（M5）。在 N5b 补证据与 N5c 通用排障推理时注入，用于约束 LLM 的每一步推断。本文件由 scripts/render_constraint_skill.py 从 rca_framework/constraints/library.py 自动生成，不要手工编辑。
---

# 光模块物理约束库

版本 `constraint-library-v3`，内容指纹 `c090f825efe2da67`，共 15 条。
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
- **实测证据**：按 lane 号配对后，L1->L2 方向的均值损耗中位数为 -0.285 dB，L2->L1 为 -0.227 dB，两个方向的中位数都是负值，物理上不可能。legacy `directional_loss` 学到的上界（3.11 / 3.42 dB）因此也不可信。
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

## 注入 prompt 的文本块

以下内容由 `render_prompt_block()` 产出，是真正进入 prompt 的原文。

```text
# 光模块物理约束（constraint-library-v3，hash c090f825efe2da67）


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
- [C2_bias_healthy_band] 健康 lane 的偏置电流在 7.2-7.8 mA。电流在此范围内说明激光器驱动正常，不要把根因归到发送端器件老化。（待专家审核）
  结构化引用契约：可用 token 前缀=drop:L1:bias:、drop:L2:bias:；effect 只能为 neutral；target 只能为 空字符串。
- [C8_tx_ok_rx_down_indicates_medium] 如果本端某 lane 发光正常而对端同一 lane 完全收不到光，说明光在路径中丢失，候选根因是光纤介质或对端接收器件。这条线索会提高光纤的可能性，但不足以定论，必须再结合双向一致性判断。（待专家审核）
  结构化引用契约：可用 token 前缀=lane:L1_to_L2:tx_ok_rx_down、lane:L2_to_L1:tx_ok_rx_down；effect 只能为 support；target 只能为 fiber。
- [C9_bidirectional_symmetry] 同一条 lane 在两个方向上同时异常，说明问题出在双向共享的部分，即光纤对或连接器，光纤是首位候选。如果只有一个方向异常，则问题在该方向的端点设备，光纤应当降位。（待专家审核）
  结构化引用契约：可用 token 前缀=lane:L1_to_L2:bidirectional_same_lane、lane:L2_to_L1:bidirectional_same_lane；effect 只能为 support；target 只能为 fiber。
```

## 待办

- 全部 15 条均为 `pending_expert_review`，需夏思博逐条确认后改为 `approved`。
- `measured` 类参数绑定当前数据集切分，合并数据集到位后必须重测。
- `C12` 与 `C13` 是数据质量问题，需要向厂商确认 lane 编号对应关系与 `serdes_snr` 量纲。
