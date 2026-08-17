# 扩充 RCA 远端复现实验

本轮使用 `expanded-expert-clean-v1`：剔除 6 条低质量 blackout 后，训练集为 122 条、
测试集为 341 条；已审核 case 使用专家标签，未审核 case 保留原标签并在
`data_contract.json` 中标为 `unreviewed`。不改 prompt、模型或 legacy 评估逻辑。
脚本使用远端已验证的本地 `DeepSeek-R1-Distill-Qwen-32B` checkpoint、自动选择两张空闲 GPU，并复用正式入口
`python -m rca_framework.cli train-evaluate`。vLLM 在实验进程内加载模型，不需要 API
server、API key 或网络连接。

## 上传

建议把整个当前仓库同步到远端 `/home/chenziang/nsdi-agent`，至少保证下列文件存在：

- `datasets/organized_rca_v2_stratified_60_40_seed42/`（只读来源与兼容检查）；
- `experiments/20260816_expanded-pattern-conflict/clean_train.jsonl`；
- `experiments/20260816_expanded-pattern-conflict/clean_expanded_test.jsonl`；
- `experiments/20260816_expanded-pattern-conflict/data_contract.json`；
- `experiments/20260816_expanded-pattern-conflict/added_case_ids.txt`；
- `experiments/20260816_expanded-pattern-conflict/added_cases_manifest.json`；
- `experiments/20260816_expanded-pattern-conflict/run_expanded_remote_experiment.sh`；
- `scripts/materialize_expanded_legacy_dataset.py`；
- 当前 `rca_framework/` 代码。

推荐结构：

```text
/home/chenziang/nsdi-agent/
  rca_framework/
  scripts/materialize_expanded_legacy_dataset.py
  datasets/organized_rca_v2_stratified_60_40_seed42/
  experiments/20260816_expanded-pattern-conflict/
```

## 环境变量

脚本有与旧实验一致的远端默认值，也可显式覆盖：

```bash
export NSDI_RCA_ROOT=/home/chenziang/nsdi-agent
export NSDI_RCA_PYTHON=/home/chenziang/miniconda3/bin/python3
export NSDI_RCA_VLLM_PYTHON=/home/chenziang/miniconda3/envs/logsy/bin/python
export NSDI_RCA_MODEL_PATH=/home/chenziang/pretrained_models/DeepSeek-R1-Distill-Qwen-32B
export NSDI_RCA_TENSOR_PARALLEL_SIZE=2
export NSDI_RCA_OUTPUT_DIR=/home/chenziang/nsdi-agent/artifacts/expanded_deepseek32b_seed42
```

上述模型路径和 TP=2 已有历史正式实验记录；不设置环境变量时脚本直接采用这两个默认值。
只有 checkpoint 被移动时才需要设置 `NSDI_RCA_MODEL_PATH`。

### GPU 自动选择

默认需要 2 张卡，并把同时满足以下条件的卡视为空闲：

- 剩余显存至少 44000 MiB；
- 已用显存不超过 2048 MiB；
- GPU 利用率不超过 10%。

空闲卡不足时每 30 秒检查一次，最多等待 3600 秒。可通过下列变量调整：

```bash
export NSDI_RCA_GPU_MIN_FREE_MB=44000
export NSDI_RCA_GPU_MAX_USED_MB=2048
export NSDI_RCA_GPU_MAX_UTIL_PCT=10
export NSDI_RCA_GPU_WAIT_SECONDS=3600
export NSDI_RCA_GPU_POLL_SECONDS=30
```

通常不要设置 `CUDA_VISIBLE_DEVICES`，让脚本自动选择。如果只允许脚本使用某几张物理卡，
可将它设为候选集合，例如 `CUDA_VISIBLE_DEVICES=2,3,4,5`；脚本仍会检查这些卡是否空闲。
32B BF16 的已验证配置是两张 RTX A6000 48GB、TP=2、显存利用率 0.85。通常不需要修改
`NSDI_RCA_TENSOR_PARALLEL_SIZE` 和空闲显存门槛。

## 执行

```bash
cd /home/chenziang/nsdi-agent
bash experiments/20260816_expanded-pattern-conflict/run_expanded_remote_experiment.sh
```

脚本默认 `NSDI_RCA_TRAIN_SIZE=122`、`NSDI_RCA_EXPECTED_TEST_SIZE=341`。如需显式指定契约：

```bash
export NSDI_RCA_TRAIN_JSONL=/home/chenziang/nsdi-agent/experiments/20260816_expanded-pattern-conflict/clean_train.jsonl
export NSDI_RCA_EXPANDED_TEST_JSONL=/home/chenziang/nsdi-agent/experiments/20260816_expanded-pattern-conflict/clean_expanded_test.jsonl
```

首次建议先做不加载权重的 dry-run。它会检查本地模型配置、vLLM/Transformers/PyTorch 环境
和空闲 GPU，但不会启动正式实验：

```bash
NSDI_RCA_OUTPUT_DIR=/home/chenziang/nsdi-agent/artifacts/expanded_deepseek32b_dryrun \
NSDI_RCA_DRY_RUN=1 \
bash experiments/20260816_expanded-pattern-conflict/run_expanded_remote_experiment.sh
```

dry-run 和正式运行必须使用不同的输出目录，因为脚本会拒绝覆盖任何已有目录。

正式脚本会把“clean 122 train + clean 341 test”物化为临时 positional dataset，随后在选中的
GPU 上加载 DeepSeek-R1-Distill-Qwen-32B 并一次性评估 341 条测试 case。它拒绝覆盖已有输出，记录实际命令、模型
preflight、git commit、运行日志和 GPU 选择前/模型启动前/进程退出后的显存快照。中断脚本时会
终止实验子进程；模型进程退出后再从进程外记录 `nvidia-smi`。

## 下载回本地

从 `NSDI_RCA_OUTPUT_DIR` 下载：

- `predictions_expanded.json`
- `evaluation_expanded.json`
- `matched_train_cases.jsonl`
- `run_manifest.json`
- `metadata.json`
- `command.txt`
- `git_commit.txt`
- `run.log`
- `nvidia_smi_before_selection.txt`
- `gpu_query_at_selection.csv`
- `nvidia_smi_before_model.txt`
- `nvidia_smi_after.txt`
- `model_preflight.json`

回传后放到本地任意目录，例如 `artifacts/expanded_deepseek32b_seed42/`，然后重建 HTML：

```bash
python3 scripts/analyze_expanded_rca_patterns.py \
  --old-data-dir datasets/organized_rca_v2_stratified_60_40_seed42 \
  --new-data-dir datasets/all_data_rca_v2_stratified_60_40_seed42 \
  --output-dir experiments/20260816_expanded-pattern-conflict \
  --train-size 126 --similarity-threshold 0.70 --graph-similarity-threshold 0.70 --top-k 5 \
  --expert-annotations /path/to/expert_label_annotations.json \
  --remote-run-dir artifacts/expanded_deepseek32b_seed42 \
  --clean-missing-old
```

重建后 `expanded_rca_pattern_analysis.html` 会补上模型预测，并可逐 case 检查模型是否跟随了冲突的历史标签。
