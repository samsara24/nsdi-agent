#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

default_vllm_python="/home/chenziang/miniconda3/envs/logsy/bin/python"
if [[ -x "$default_vllm_python" ]]; then
  python_bin="${VLLM_PYTHON:-${PYTHON_BIN:-$default_vllm_python}}"
else
  python_bin="${VLLM_PYTHON:-${PYTHON_BIN:-python}}"
fi
model_path="${MODEL_PATH:-/home/chenziang/pretrained_models/DeepSeek-R1-Distill-Qwen-32B}"
data_dir="${DATA_DIR:-$repo_root/datasets/filtered_rule_temporal_2025_06_09_v1}"
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_dir="${OUTPUT_DIR:-$repo_root/artifacts/filtered_rule_temporal_${run_stamp}}"
max_gpus="${MAX_GPUS:-4}"
min_free_mb="${MIN_GPU_FREE_MB:-44000}"
max_used_mb="${MAX_GPU_USED_MB:-2048}"
max_util="${MAX_GPU_UTILIZATION:-10}"
wait_seconds="${GPU_WAIT_SECONDS:-3600}"
poll_seconds="${GPU_POLL_SECONDS:-30}"
max_model_len="${MAX_MODEL_LEN:-32768}"
max_new_tokens="${MAX_NEW_TOKENS:-16384}"

command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required" >&2; exit 2; }
test -x "$(command -v "$python_bin")" || { echo "python not found: $python_bin" >&2; exit 2; }
test -f "$data_dir/_metadata/manifest.json" || { echo "manifest not found: $data_dir" >&2; exit 2; }
test -d "$model_path" || { echo "model not found: $model_path" >&2; exit 2; }
test ! -e "$output_dir" || { echo "output already exists: $output_dir" >&2; exit 2; }

model_config="$model_path/config.json"
test -f "$model_config" || { echo "model config not found: $model_config" >&2; exit 2; }
read -r attention_heads hidden_size < <("$python_bin" -c 'import json,sys; c=json.load(open(sys.argv[1])); print(c.get("num_attention_heads",0), c.get("hidden_size",0))' "$model_config")

eligible_gpu_ids() {
  nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits |
    awk -F, -v free="$min_free_mb" -v used="$max_used_mb" -v util="$max_util" '
      {gsub(/ /, "", $0); if ($2 <= used && $3 >= free && $4 <= util) print $1}'
}

deadline=$(( $(date +%s) + wait_seconds ))
while true; do
  mapfile -t gpu_ids < <(eligible_gpu_ids)
  if (( ${#gpu_ids[@]} > 0 )); then break; fi
  if (( $(date +%s) >= deadline )); then echo "no idle GPU before timeout" >&2; exit 3; fi
  echo "waiting for an idle GPU..."
  sleep "$poll_seconds"
done

available=${#gpu_ids[@]}
(( max_gpus < available )) && available=$max_gpus
tp_size=0
if [[ -n "${TENSOR_PARALLEL_SIZE:-}" ]]; then
  tp_size="$TENSOR_PARALLEL_SIZE"
else
  for candidate in 4 3 2 1; do
    if (( candidate <= available && attention_heads % candidate == 0 && hidden_size % candidate == 0 )); then
      tp_size=$candidate
      break
    fi
  done
fi
(( tp_size >= 1 && tp_size <= ${#gpu_ids[@]} )) || { echo "invalid tensor parallel size: $tp_size" >&2; exit 3; }
selected=("${gpu_ids[@]:0:tp_size}")
cuda_visible="$(IFS=,; echo "${selected[*]}")"

mkdir -p "$output_dir"
trap 'nvidia-smi > "$output_dir/nvidia_smi_after.txt" 2>&1 || true' EXIT
nvidia-smi > "$output_dir/nvidia_smi_before.txt"
{
  echo "CUDA_VISIBLE_DEVICES=$cuda_visible"
  echo "tensor_parallel_size=$tp_size"
  echo "attention_heads=$attention_heads"
  echo "hidden_size=$hidden_size"
  echo "model_path=$model_path"
  echo "data_dir=$data_dir"
  echo "max_model_len=$max_model_len"
  echo "max_new_tokens=$max_new_tokens"
  echo "max_attempts=1"
} > "$output_dir/gpu_selection.txt"

cmd=(
  "$python_bin" scripts/run_filtered_rule_temporal_experiment.py
  --data-dir "$data_dir"
  --output-dir "$output_dir/run"
  --model-path "$model_path"
  --tensor-parallel-size "$tp_size"
  --dtype bfloat16
  --max-model-len "$max_model_len"
  --max-new-tokens "$max_new_tokens"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}"
  --max-attempts 1
)
[[ "${DISABLE_CUSTOM_ALL_REDUCE:-1}" == "1" ]] && cmd+=(--disable-custom-all-reduce)
[[ "${ENFORCE_EAGER:-1}" == "1" ]] && cmd+=(--enforce-eager)
[[ -n "${TARGET_SELECTIVE_RISK:-}" ]] && cmd+=(--target-selective-risk "$TARGET_SELECTIVE_RISK")

printf '%q ' env CUDA_VISIBLE_DEVICES="$cuda_visible" NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}" VLLM_USE_V1="${VLLM_USE_V1:-1}" VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}" "${cmd[@]}" > "$output_dir/command.txt"
printf '\n' >> "$output_dir/command.txt"

set +e
CUDA_VISIBLE_DEVICES="$cuda_visible" \
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" \
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}" \
VLLM_USE_V1="${VLLM_USE_V1:-1}" \
VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}" \
"${cmd[@]}" 2>&1 | tee "$output_dir/console.log"
status=${PIPESTATUS[0]}
set -e
nvidia-smi > "$output_dir/nvidia_smi_after.txt"
(( status == 0 )) || exit "$status"

for required in \
  "$output_dir/run/run_manifest.json" \
  "$output_dir/run/summary.json" \
  "$output_dir/run/test_all_data/html/index.html" \
  "$output_dir/run/test_rule1_channel_not_4/html/index.html"; do
  test -s "$required" || { echo "missing artifact: $required" >&2; exit 4; }
done
echo "completed: $output_dir"
