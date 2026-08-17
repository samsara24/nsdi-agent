#!/usr/bin/env bash
set -euo pipefail

ROOT="${NSDI_RCA_ROOT:-/home/chenziang/nsdi-agent}"
BASE_PYTHON="${NSDI_RCA_PYTHON:-/home/chenziang/miniconda3/bin/python3}"
VLLM_PYTHON="${NSDI_RCA_VLLM_PYTHON:-/home/chenziang/miniconda3/envs/logsy/bin/python}"
MODEL_PATH="${NSDI_RCA_MODEL_PATH:-/home/chenziang/pretrained_models/DeepSeek-R1-Distill-Qwen-32B}"
DATA_DIR="${NSDI_RCA_DATA_DIR:-datasets/organized_rca_v2_stratified_60_40_seed42}"
TRAIN_SIZE="${NSDI_RCA_TRAIN_SIZE:-126}"
RUN_VLLM="${NSDI_RCA_RUN_VLLM:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"

cd "$ROOT"

timestamp="$(date +%Y%m%d_%H%M%S)"
run_root="${NSDI_RCA_OUTPUT_ROOT:-artifacts/main_experiment_${timestamp}}"
baseline_out="${run_root}/baseline_none"
llm_out="${run_root}/deepseek32b_vllm"
archive_manifest="archive/organized_data_source_manifest_${timestamp}.json"

echo "[nsdi-rca] root: $ROOT"
echo "[nsdi-rca] data: $DATA_DIR"
echo "[nsdi-rca] output: $run_root"
mkdir -p "$run_root"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "[nsdi-rca] dataset not found; preparing organized stratified split"
  PYTHONPATH="$ROOT" "$BASE_PYTHON" scripts/prepare_organized_stratified.py \
    --input-dir organized_data \
    --output-dir "$DATA_DIR" \
    --archive-manifest "$archive_manifest" \
    --train-ratio 0.6 \
    --seed 42
else
  echo "[nsdi-rca] reusing existing dataset"
fi

echo "[nsdi-rca] running deterministic baseline"
PYTHONPATH="$ROOT" "$BASE_PYTHON" -m rca_framework.cli train-evaluate \
  --data-dir "$DATA_DIR" \
  --train-size "$TRAIN_SIZE" \
  --output-dir "$baseline_out" \
  --backend none

if [[ "$RUN_VLLM" == "1" ]]; then
  if [[ ! -x "$VLLM_PYTHON" ]]; then
    echo "[nsdi-rca] vLLM python not executable: $VLLM_PYTHON" >&2
    exit 2
  fi
  if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[nsdi-rca] model path not found: $MODEL_PATH" >&2
    exit 2
  fi
  echo "[nsdi-rca] running DeepSeek-32B vLLM evaluation on GPUs: $CUDA_VISIBLE_DEVICES"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" PYTHONPATH="$ROOT" "$VLLM_PYTHON" -m rca_framework.cli train-evaluate \
    --data-dir "$DATA_DIR" \
    --train-size "$TRAIN_SIZE" \
    --output-dir "$llm_out" \
    --backend vllm \
    --model-path "$MODEL_PATH" \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --max-new-tokens 512 \
    --dtype bfloat16 \
    --enforce-eager \
    --disable-custom-all-reduce
else
  echo "[nsdi-rca] skipping vLLM evaluation because NSDI_RCA_RUN_VLLM=$RUN_VLLM"
fi

echo "[nsdi-rca] summary"
PYTHONPATH="$ROOT" "$BASE_PYTHON" scripts/summarize_runs.py --artifacts "$run_root"
