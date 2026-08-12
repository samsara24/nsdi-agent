# 专家模型

# 专家规则方案总结



## 1\. 总体定界入口



输入先经过 `DiagnosticLogProcessor.process_diagnostic_data()` 标准化，然后进入故障定位主流程。



主流程先检查两端端口状态：



|local\_port\_status|remote\_port\_status|直接定界|原因|
|---|---|---|---|
|0|1|local|本端端口已 down，优先检查|
|1|0|remote|对端端口已 down，优先检查|
|0|0|local|两端端口均 down，无法诊断，优先检查本端|
|1|1|进入专家规则 \+ RF 模型|两端端口均可用，继续定位|



端口状态不是直接读输入字段，而是预处理器根据 `txpower` 与 `rxpower` 判断：



• 对每一端分别检查 `txpower`、`rxpower`。



• 若某指标缺失，或该端该指标没有任何一个 lane 满足 `v > down_threshold`，则该指标记为异常。



• 两个指标都异常时：`port_status[side] = 0`。



• 否则：`port_status[side] = 1`。



• 这里实现实际只检查 `txpower`、`rxpower` 两个指标；注释里“txpower、rxpower三个指标”明显不严谨。





## 2\. 专家规则输入数据预处理



预处理器从 `work_order_result` 中抽取本端/对端数据，形成标准结构。



### 2\.1 多 lane 指标



原始字段：



• `Bias`



• `RxPower`



• `TxPower`



• `MediaSNR`



• `HostSNR`



• `PhySds SNR`





重命名为：



|原始字段|标准字段|
|---|---|
|Bias|bias|
|RxPower|rxpower|
|TxPower|txpower|
|MediaSNR|media\_snr|
|HostSNR|host\_snr|
|PhySds SNR|serdes\_snr|



多 lane 字段按 lane 编号提取，例如 `Lane0`、`Lane1`，最终形态类似：



```Python
{
  "rxpower": {
    "local": {"0": -1.2, "1": -1.1},
    "remote": {"0": -1.3, "1": -1.2}
  }
}
```



### 2\.2 单值状态字段



单值字段包括：



• `RxLOL`



• `TxLOL`



• `TxLOS`



• `RxLOS`



• `Lane number`





结构为：



```Python
{
  "RxLOS": {
    "local": "...",
    "remote": "..."
  }
}
```



### 2\.3 特殊值处理



• 字符串 `"不涉及"` 会递归替换为 `None`。



• 字符串 `"不合规--xxx"` 会抽取 `xxx` 并尝试转为 int/float。



• 多 lane 指标按 lane 编号排序。



• `host_snr` 特殊后处理：如果某端所有 lane 都没有有效值，即没有任何 `v > 0`，则该端 `host_snr` 被置为 `None`。





## 3\. 专家规则的异常判断指标与阈值



专家规则只遍历 `ANOMALY_TYPE_AND_THRESHOLD_SETTING` 中配置的指标：



• `rxpower`



• `txpower`



• `host_snr`



• `media_snr`



• `serdes_snr`





注意：`bias` 虽然在预处理和 RF 特征里出现，但专家规则阈值表里没有 `bias`，所以专家规则不会直接用 `bias` 定界。



### 3\.1 异常类型



每个指标最多返回一个异常，检测顺序是强制短路的：



1. `lane_down`

2. `low_value`

3. `high_value`

4. `lane_diff`

    

只要前面的异常命中，后面的异常不再检查。



例如某个指标同时满足低值和 lane\_diff，最终只会被记录为 `low_value`。



### 3\.2 异常类型判定逻辑



|异常类型|判定逻辑|
|---|---|
|lane\_down|任一 lane 值等于 down 阈值|
|low\_value|任一 lane 值 `< low_threshold`|
|high\_value|任一 lane 值 `> high_threshold`|
|lane\_diff|`max(lane_values) - min(lane_values) > lane_diff_threshold`|



### 3\.3 具体阈值



|指标|lane\_down|low\_value|high\_value|lane\_diff|
|---|---|---|---|---|
|rxpower|\-40|\-2\.5|4\.6|1|
|txpower|\-40|\-2\.5|2\.5|1\.3|
|host\_snr|0|22\.8|27\.5|2\.5|
|media\_snr|0|22\.4|28\.7|3|
|serdes\_snr|0|458750|947750|230000|



## 4\. 异常级别优先级



`AnomalyLevel` 定义为数值越小优先级越高：



|异常级别|数值|含义|
|---|---|---|
|LANE\_DOWN|0|lane down，最高|
|LOW\_VALUE|1|低值|
|HIGH\_VALUE|1|高值，与低值同级|
|LANE\_DIFF|2|多 lane 差异过大，最低|



## 5\. 专家规则模式匹配



每一端分别做异常检测，然后调用 `FaultPatternMatcher.match_pattern(side, anomalies)`。



专家规则的输出类别只有三类：



• `local`



• `remote`



• `fiber`





### 5\.1 模式 1：TxPowerLaneDownPattern



条件：



```Plain Text
该端 txpower 存在，且 txpower.anomaly_type == lane_down
```



定界：



```Plain Text
故障位置 = 异常所在端
```



优先级：



```Plain Text
0
```



这是最高优先级规则。



### 5\.2 模式 2：MultiMetricAnomalyPattern



条件：



```Plain Text
该端同时存在：
- serdes_snr 异常
- media_snr 异常
- rxpower 异常
```



定界：



```Plain Text
故障位置 = 异常所在端的对端
```



优先级：



```Plain Text
1
```



必须注意：代码注释写了“如果 rxpower 有 lane\_down 异常，故障位置在对端”，但实际实现并没有检查 `rxpower.anomaly_type == lane_down`。实际代码只要求 `serdes_snr`、`media_snr`、`rxpower` 三个指标都在 anomalies 里。也就是说：



```Plain Text
rxpower 任意异常 + media_snr 任意异常 + serdes_snr 任意异常
```



都会触发该规则，并定界到对端。



这是一个规则语义与注释不一致的点，后续如果做评审必须单独指出。



### 5\.3 模式 3：SingleMetricPattern



单指标异常模式按指标配置故障方向。



|指标|基础优先级|故障位置映射|
|---|---|---|
|host\_snr|2|异常所在端|
|serdes\_snr|3|异常所在端|
|media\_snr|4|异常所在端的对端|
|rxpower|5|异常所在端的对端|
|txpower|6|异常所在端|



注意：`FaultPatternMatcher` 初始化列表里 `rxpower` 写在 `serdes_snr` 前面，但最终会按 priority 排序，所以真实优先级仍是：



```Plain Text
txpower lane_down
> 多指标异常
> host_snr
> serdes_snr
> media_snr
> rxpower
> txpower
```



单指标规则的最终 priority 是字符串拼接：



```Plain Text
f"{rule_priority.value}{anomaly.level}"
```



因此示例为：



|规则|lane\_down|low/high|lane\_diff|
|---|---|---|---|
|host\_snr|20|21|22|
|serdes\_snr|30|31|32|
|media\_snr|40|41|42|
|rxpower|50|51|52|
|txpower|60|61|62|



## 6\. 两端结果裁决流程



专家模型对 local 和 remote 分别诊断，然后合并裁决。



### 6\.1 单端诊断



对每一端：



1. 解析该端指标。

2. 按阈值表检测异常。

3. 得到 `{metric_name: AnomalyInfo}`。

4. 用 `FaultPatternMatcher` 匹配所有规则。

5. 若多个规则命中，按 priority 升序，取最高优先级。

    

### 6\.2 双端合并



得到 local 和 remote 两端的诊断结果后：



1. 过滤掉 `fault_location is None` 的结果。

    

2. 按 priority 升序排序。

    

3. 如果两端都没有有效结果：

    

    • 返回 `local`

    

    • 原因：`两端指标无明显异常`

    

    • priority = `8`

    

    

4. 如果两端都有有效结果，且：

    

    • 两端预测的 `fault_location` 不同；

    

    • 两端 priority 相同；

    

    

    则返回：

    

    • `fiber`

    

    • 原因：`两端均异常`

    

    • priority = `7`

    

    

5. 其他情况：

    

    • 返回 priority 最小的结果。

    

    

## 7\. 全局专家规则优先级表



|优先级|规则|定界|
|---|---|---|
|0|txpower lane\_down|异常所在端|
|1|serdes\_snr \+ media\_snr \+ rxpower 同端同时异常|异常所在端的对端|
|20/21/22|host\_snr 单指标异常|异常所在端|
|30/31/32|serdes\_snr 单指标异常|异常所在端|
|40/41/42|media\_snr 单指标异常|异常所在端的对端|
|50/51/52|rxpower 单指标异常|异常所在端的对端|
|60/61/62|txpower 单指标异常|异常所在端|
|7|两端异常且不同定界、同优先级|fiber|
|8|两端无明显异常|local|



这里有一个设计问题：`BOTH_ANOMALY = 7` 和 `NO_ANOMALY = 8` 是枚举优先级，但专家合并阶段直接返回，不再参与和其他规则排序，所以它们不是普通匹配规则优先级，而是兜底裁决类别。



## 8\. 需要警惕的专家规则问题



1. priority 是字符串排序，不是数值排序。

当前 priority 都是一位前缀，字符串排序暂时等价于数值排序；但如果未来规则优先级出现 10、11，就会出现 `"10" < "2"` 这类排序陷阱。



2. 多指标异常规则注释与实现不一致。

注释强调 rxpower lane\_down，但代码只要求 rxpower 任意异常。



3. 每个指标只保留一个异常。

因为 `_detect_anomaly()` 是短路返回，所以无法表达“同一指标既低值又 lane\_diff”的复合异常。



4. 无异常时默认返回 local。

这不是“定位到了 local”，而是一个兜底默认值。业务解释上必须谨慎，否则会造成 local 偏置。



5. 两端同优先级但定界相同，不会返回 fiber。

只有“两端 fault\_location 不同且 priority 相同”才返回 fiber。











## 华为word

\(1\)        专家决策树模型

基于专家业务经验构建规则决策树，

指标异常优先级：

        down异常：优先级0

        指标值异常：优先级1

        指标离群异常：优先级2

故障定位模式优先级：

        txpower down异常，优先级0

        mediaSNR、serdesSNR、rxpower组合异常（全部异常），优先级1

        hostSNR异常，优先级2

        serdesSNR异常，优先级3

        mediaSNR异常，优先级4

        rxpower异常，优先级5

        txpower非down异常，优先级6

单侧光模块定位逻辑如下图，定位优先级= \{故障定位模式优先级\}\{指标异常优先级\}，例如rxpower值离群异常，其他指标正常，则优先级为‘52’



图 单侧指标故障定位决策树

结合两端光模块的定位结果和优先级给出最终光链路故障定位结果，定位流程如图：

        若一端存在异常，另一端正常，则按照异常端定位为准；

        若两端均存在异常，则比较两端定位的优先级，以高优先级定位结果为准；若两端定位优先级相同，但定位结果不同，则为光纤故障；

        若两端均正常，则优先反馈本端故障



