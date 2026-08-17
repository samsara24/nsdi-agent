# Loop1 假设（启动时）

## 上一轮来源

`experiments/` 尚无正式归档；以上一轮正式产物
`artifacts/offline_sop_llm_l2fixed_deepseek32b_seed42_promptv6` 提炼 bad case，
并写入 `experiments/_seed_from_promptv6/`。

上一轮分支级错误 21 条：

- N5a_pure 4：完全相同 signature 复用了不同标签历史 → 疑似 label / irreducible
- N5b_minor_gap 8：缺失多为 `level:*`，当时未判关键，直接复用历史多数
- N5c 9：大量硬猜 fiber

## 本轮优化靶心

物理约束下的关键证据判定（兼修 N5c fiber prompt）。

## 假设

1. P10 用笼统 `level:` / `drop:` / `status:` 前缀，会把发送侧分档误当成接收症状关键证据。
2. 量测契约与 P4（正常带内发送电平非归因）不该进入「缺失则关键」门禁。
3. 收窄 P10 + 只保留归因类物理约束做 key-when-missing 后：
   - 接收类缺失（rxpower / media_snr / RxLOS）会正确进 LLM 仲裁；
   - 仅缺 txpower 分档的 case 不再被误抬成关键。
4. N5c / N5b prompt 明确 C20：无双向已发光对称证据不得输出 fiber。

## 允许改动类别

- `bug_fix`：关键证据判定语义与 P10 前缀过宽
- `evidence_graph` / 约束 schema：P10 applies_to 收窄
- `llm_prompt`：v10 / diagnose-v3 的 fiber 硬规则

## 运行

- 首次用系统 python 失败：`ModuleNotFoundError: vllm`
- 已用 `/home/chenziang/miniconda3/envs/logsy` 重跑：
  `artifacts/loop1_n5b_key_evidence_v10_20260812_b`
