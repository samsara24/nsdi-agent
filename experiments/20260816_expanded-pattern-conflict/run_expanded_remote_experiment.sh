#!/usr/bin/env bash
set -euo pipefail

# One-command remote runner for the cleaned/adjudicated expanded 341-case test set. vLLM is
# embedded by the dual-similarity experiment runner, so DeepSeek-R1-Distill-Qwen-32B is loaded locally
# process; no API server, API key, or Internet connection is required.

ROOT="${NSDI_RCA_ROOT:-/home/chenziang/nsdi-agent}"
BASE_PYTHON="${NSDI_RCA_PYTHON:-/home/chenziang/miniconda3/bin/python3}"
VLLM_PYTHON="${NSDI_RCA_VLLM_PYTHON:-/home/chenziang/miniconda3/envs/logsy/bin/python}"
MODEL_PATH="${NSDI_RCA_MODEL_PATH:-/home/chenziang/pretrained_models/DeepSeek-R1-Distill-Qwen-32B}"
TRAIN_JSONL="${NSDI_RCA_TRAIN_JSONL:-$ROOT/experiments/20260816_expanded-pattern-conflict/clean_train.jsonl}"
EXPANDED_TEST_JSONL="${NSDI_RCA_EXPANDED_TEST_JSONL:-$ROOT/experiments/20260816_expanded-pattern-conflict/clean_expanded_test.jsonl}"
DATA_CONTRACT="${NSDI_RCA_DATA_CONTRACT:-$ROOT/experiments/20260816_expanded-pattern-conflict/data_contract.json}"
OUTPUT_DIR="${NSDI_RCA_OUTPUT_DIR:-$ROOT/artifacts/expanded_deepseek32b_$(date +%Y%m%d_%H%M%S)}"
TRAIN_SIZE="${NSDI_RCA_TRAIN_SIZE:-122}"
EXPECTED_TEST_SIZE="${NSDI_RCA_EXPECTED_TEST_SIZE:-341}"

# The validated server configuration is two RTX A6000 48-GB GPUs in BF16.
TENSOR_PARALLEL_SIZE="${NSDI_RCA_TENSOR_PARALLEL_SIZE:-2}"
GPU_MIN_FREE_MB="${NSDI_RCA_GPU_MIN_FREE_MB:-44000}"
GPU_MAX_USED_MB="${NSDI_RCA_GPU_MAX_USED_MB:-2048}"
GPU_MAX_UTIL_PCT="${NSDI_RCA_GPU_MAX_UTIL_PCT:-10}"
GPU_WAIT_SECONDS="${NSDI_RCA_GPU_WAIT_SECONDS:-3600}"
GPU_POLL_SECONDS="${NSDI_RCA_GPU_POLL_SECONDS:-30}"
REQUESTED_CUDA="${CUDA_VISIBLE_DEVICES:-}"
GPU_MEMORY_UTILIZATION="${NSDI_RCA_GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${NSDI_RCA_MAX_MODEL_LEN:-12288}"
MAX_NEW_TOKENS="${NSDI_RCA_MAX_NEW_TOKENS:-512}"
DRY_RUN="${NSDI_RCA_DRY_RUN:-0}"

for value in "$TRAIN_SIZE" "$EXPECTED_TEST_SIZE" "$TENSOR_PARALLEL_SIZE" "$GPU_MIN_FREE_MB" "$GPU_MAX_USED_MB" \
  "$GPU_MAX_UTIL_PCT" "$GPU_WAIT_SECONDS" "$GPU_POLL_SECONDS" "$MAX_MODEL_LEN" "$MAX_NEW_TOKENS"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "expected a non-negative integer, got: $value" >&2; exit 2; }
done
(( TENSOR_PARALLEL_SIZE > 0 )) || { echo "tensor parallel size must be positive" >&2; exit 2; }
(( GPU_POLL_SECONDS > 0 )) || { echo "GPU poll interval must be positive" >&2; exit 2; }

for path in "$ROOT"; do
  [[ -d "$path" ]] || { echo "missing directory: $path" >&2; exit 2; }
done
for path in "$TRAIN_JSONL" "$EXPANDED_TEST_JSONL" "$DATA_CONTRACT" "$ROOT/scripts/run_expanded_dual_experiment.py"; do
  [[ -f "$path" ]] || { echo "missing file: $path" >&2; exit 2; }
done
[[ -x "$BASE_PYTHON" ]] || { echo "base python is not executable: $BASE_PYTHON" >&2; exit 2; }
[[ -x "$VLLM_PYTHON" ]] || { echo "vLLM python is not executable: $VLLM_PYTHON" >&2; exit 2; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi is required" >&2; exit 2; }
[[ ! -e "$OUTPUT_DIR" ]] || { echo "refusing to overwrite output: $OUTPUT_DIR" >&2; exit 2; }

mkdir -p "$OUTPUT_DIR"
LOG="$OUTPUT_DIR/run.log"
exec > >(tee -a "$LOG") 2>&1
cd "$ROOT"

timestamp() { date --iso-8601=seconds 2>/dev/null || date; }

[[ -n "$MODEL_PATH" && -d "$MODEL_PATH" ]] || {
  echo "model directory was not found: $MODEL_PATH" >&2
  echo "Set NSDI_RCA_MODEL_PATH if the local checkpoint was moved." >&2
  exit 2
}
if [[ "${MODEL_PATH,,}" != *deepseek-r1-distill-qwen-32b* && "${NSDI_RCA_ALLOW_OTHER_MODEL:-0}" != "1" ]]; then
  echo "refusing an unexpected model path: $MODEL_PATH" >&2
  echo "Set NSDI_RCA_ALLOW_OTHER_MODEL=1 only for an intentional model change." >&2
  exit 2
fi
[[ -f "$MODEL_PATH/config.json" ]] || { echo "model config is missing: $MODEL_PATH/config.json" >&2; exit 2; }
ACTUAL_TEST_SIZE="$(NSDI_TEST_JSONL="$EXPANDED_TEST_JSONL" "$BASE_PYTHON" -c 'import os,pathlib; print(sum(bool(line.strip()) for line in pathlib.Path(os.environ["NSDI_TEST_JSONL"]).open(encoding="utf-8")))')"
ACTUAL_TRAIN_SIZE="$(NSDI_TRAIN_JSONL="$TRAIN_JSONL" "$BASE_PYTHON" -c 'import os,pathlib; print(sum(bool(line.strip()) for line in pathlib.Path(os.environ["NSDI_TRAIN_JSONL"]).open(encoding="utf-8")))')"
[[ "$ACTUAL_TRAIN_SIZE" == "$TRAIN_SIZE" ]] || {
  echo "clean train size mismatch: expected $TRAIN_SIZE, got $ACTUAL_TRAIN_SIZE" >&2
  exit 2
}
[[ "$ACTUAL_TEST_SIZE" == "$EXPECTED_TEST_SIZE" ]] || {
  echo "expanded test size mismatch: expected $EXPECTED_TEST_SIZE, got $ACTUAL_TEST_SIZE" >&2
  exit 2
}

echo "[expanded-rca] start: $(timestamp)"
echo "[expanded-rca] root: $ROOT"
echo "[expanded-rca] model: $MODEL_PATH"
echo "[expanded-rca] clean train: $TRAIN_JSONL"
echo "[expanded-rca] expanded test: $EXPANDED_TEST_JSONL"
echo "[expanded-rca] output: $OUTPUT_DIR"
echo "[expanded-rca] GPU policy: TP=$TENSOR_PARALLEL_SIZE, free>=${GPU_MIN_FREE_MB}MiB, used<=${GPU_MAX_USED_MB}MiB, util<=${GPU_MAX_UTIL_PCT}%"
[[ -z "$REQUESTED_CUDA" ]] || echo "[expanded-rca] GPU candidates restricted by CUDA_VISIBLE_DEVICES=$REQUESTED_CUDA"

git rev-parse HEAD > "$OUTPUT_DIR/git_commit.txt" 2>/dev/null || printf 'unknown\n' > "$OUTPUT_DIR/git_commit.txt"
nvidia-smi > "$OUTPUT_DIR/nvidia_smi_before_selection.txt"

SELECTED_GPUS=""
GPU_QUERY_ROWS=""
select_idle_gpus() {
  local selected=()
  local idx uuid total used free util
  GPU_QUERY_ROWS="$(nvidia-smi \
    --query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits | tr -d ' ')"
  while IFS=',' read -r idx uuid total used free util; do
    [[ -n "$idx" ]] || continue
    [[ "$idx" =~ ^[0-9]+$ && "$total" =~ ^[0-9]+$ && "$used" =~ ^[0-9]+$ \
      && "$free" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]] || continue
    if [[ -n "$REQUESTED_CUDA" && ",${REQUESTED_CUDA}," != *",${idx},"* ]]; then
      continue
    fi
    if (( free >= GPU_MIN_FREE_MB && used <= GPU_MAX_USED_MB && util <= GPU_MAX_UTIL_PCT )); then
      selected+=("$idx")
    fi
    if (( ${#selected[@]} == TENSOR_PARALLEL_SIZE )); then
      break
    fi
  done <<< "$GPU_QUERY_ROWS"
  if (( ${#selected[@]} == TENSOR_PARALLEL_SIZE )); then
    SELECTED_GPUS="$(IFS=,; printf '%s' "${selected[*]}")"
    return 0
  fi
  SELECTED_GPUS=""
  return 1
}

wait_started="$(date +%s)"
while ! select_idle_gpus; do
  now="$(date +%s)"
  elapsed=$((now - wait_started))
  echo "[expanded-rca] insufficient idle GPUs after ${elapsed}s; current rows:"
  printf '%s\n' "$GPU_QUERY_ROWS"
  if (( elapsed >= GPU_WAIT_SECONDS )); then
    echo "timed out waiting for $TENSOR_PARALLEL_SIZE idle GPUs" >&2
    exit 3
  fi
  sleep "$GPU_POLL_SECONDS"
done
export CUDA_VISIBLE_DEVICES="$SELECTED_GPUS"
printf '%s\n' "$GPU_QUERY_ROWS" > "$OUTPUT_DIR/gpu_query_at_selection.csv"
nvidia-smi > "$OUTPUT_DIR/nvidia_smi_before_model.txt"
echo "[expanded-rca] selected physical GPUs: $CUDA_VISIBLE_DEVICES"

# Offline preflight: inspect local config without allocating model weights.
NSDI_PREFLIGHT_MODEL="$MODEL_PATH" NSDI_PREFLIGHT_MAX_LEN="$MAX_MODEL_LEN" "$VLLM_PYTHON" - <<'PY' > "$OUTPUT_DIR/model_preflight.json"
import json, os
import torch, transformers, vllm
from transformers import AutoConfig

path = os.environ["NSDI_PREFLIGHT_MODEL"]
config = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)
requested_max_len = int(os.environ["NSDI_PREFLIGHT_MAX_LEN"])
model_max_len = getattr(config, "max_position_embeddings", None)
if model_max_len is not None and requested_max_len > int(model_max_len):
    raise ValueError(
        f"requested max_model_len {requested_max_len} exceeds model capacity {model_max_len}"
    )
print(json.dumps({
    "model_path": path,
    "model_type": getattr(config, "model_type", None),
    "architectures": getattr(config, "architectures", None),
    "hidden_size": getattr(config, "hidden_size", None),
    "num_hidden_layers": getattr(config, "num_hidden_layers", None),
    "model_max_position_embeddings": model_max_len,
    "requested_max_model_len": requested_max_len,
    "torch_version": torch.__version__,
    "transformers_version": transformers.__version__,
    "vllm_version": vllm.__version__,
    "cuda_device_count_visible": torch.cuda.device_count(),
}, ensure_ascii=False, indent=2))
PY
cat "$OUTPUT_DIR/model_preflight.json"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[expanded-rca] building all routed prompts and checking tokenizer context lengths"
  env PYTHONPATH="$ROOT" "$VLLM_PYTHON" scripts/run_expanded_dual_experiment.py \
    --train-jsonl "$TRAIN_JSONL" \
    --test-jsonl "$EXPANDED_TEST_JSONL" \
    --data-contract "$DATA_CONTRACT" \
    --output-dir "$OUTPUT_DIR/preflight_unused" \
    --backend none \
    --model-path "$MODEL_PATH" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --prompt-preflight-only > "$OUTPUT_DIR/prompt_context_preflight.json"
  cat "$OUTPUT_DIR/prompt_context_preflight.json"
  echo "[expanded-rca] dry-run passed; prompt context fits and model weights were not loaded"
  exit 0
fi

RUN_OUTPUT="$OUTPUT_DIR/deepseek32b_vllm"
COMMAND=("$VLLM_PYTHON" scripts/run_expanded_dual_experiment.py
  --train-jsonl "$TRAIN_JSONL"
  --test-jsonl "$EXPANDED_TEST_JSONL"
  --data-contract "$DATA_CONTRACT"
  --output-dir "$RUN_OUTPUT"
  --backend vllm
  --model-path "$MODEL_PATH"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --dtype bfloat16
  --enforce-eager
  --disable-custom-all-reduce)

# Past runs on this PCIe server required these NCCL settings. Each remains
# overridable for a machine with verified NVLink/IB support.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"

printf '%q ' env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
  "NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE" "NCCL_IB_DISABLE=$NCCL_IB_DISABLE" \
  "PYTHONPATH=$ROOT" "${COMMAND[@]}" > "$OUTPUT_DIR/command.txt"
printf '\n' >> "$OUTPUT_DIR/command.txt"

EXPERIMENT_PID=""
AFTER_SNAPSHOT_WRITTEN=0
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "$EXPERIMENT_PID" ]] && kill -0 "$EXPERIMENT_PID" 2>/dev/null; then
    echo "[expanded-rca] terminating experiment process $EXPERIMENT_PID"
    kill -TERM "$EXPERIMENT_PID" 2>/dev/null || true
    wait "$EXPERIMENT_PID" 2>/dev/null || true
  fi
  if [[ "$AFTER_SNAPSHOT_WRITTEN" != "1" ]]; then
    nvidia-smi > "$OUTPUT_DIR/nvidia_smi_after.txt" 2>&1 || true
  fi
  echo "[expanded-rca] exit status: $status at $(timestamp)"
  exit "$status"
}
trap cleanup EXIT INT TERM

echo "[expanded-rca] loading DeepSeek-R1-Distill-Qwen-32B with vLLM for dual-similarity + executable SOP evaluation"
env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" PYTHONPATH="$ROOT" \
  NCCL_P2P_DISABLE="$NCCL_P2P_DISABLE" NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
  VLLM_USE_V1="$VLLM_USE_V1" VLLM_ENABLE_V1_MULTIPROCESSING="$VLLM_ENABLE_V1_MULTIPROCESSING" \
  "${COMMAND[@]}" &
EXPERIMENT_PID=$!
wait "$EXPERIMENT_PID"
EXPERIMENT_PID=""

# Process exit is the process-external GPU release boundary.
nvidia-smi > "$OUTPUT_DIR/nvidia_smi_after.txt"
AFTER_SNAPSHOT_WRITTEN=1

for file in predictions.json summary.json run_manifest.json report.html; do
  [[ -s "$RUN_OUTPUT/$file" ]] || { echo "missing experiment output: $RUN_OUTPUT/$file" >&2; exit 4; }
done
NSDI_EVAL_SUMMARY="$RUN_OUTPUT/summary.json" NSDI_EXPECTED_TEST_SIZE="$EXPECTED_TEST_SIZE" \
"$BASE_PYTHON" -c 'import json,os,pathlib; value=json.loads(pathlib.Path(os.environ["NSDI_EVAL_SUMMARY"]).read_text()); actual=int(value.get("test_size",-1)); expected=int(os.environ["NSDI_EXPECTED_TEST_SIZE"]); assert actual==expected, f"evaluation test_size mismatch: expected {expected}, got {actual}"'
cp "$RUN_OUTPUT/predictions.json" "$OUTPUT_DIR/predictions_expanded.json"
cp "$RUN_OUTPUT/summary.json" "$OUTPUT_DIR/evaluation_expanded.json"
cp "$RUN_OUTPUT/run_manifest.json" "$OUTPUT_DIR/run_manifest.json"
cp "$RUN_OUTPUT/report.html" "$OUTPUT_DIR/report.html"

NSDI_MATCH_INPUT="$RUN_OUTPUT/predictions.json" NSDI_MATCH_OUTPUT="$OUTPUT_DIR/matched_train_cases.jsonl" \
"$BASE_PYTHON" -c 'import json,os,pathlib; rows=json.loads(pathlib.Path(os.environ["NSDI_MATCH_INPUT"]).read_text()); out=pathlib.Path(os.environ["NSDI_MATCH_OUTPUT"]); out.write_text("".join(json.dumps({"case_id":r.get("case_id"),"actual_label":r.get("actual_label"),"branch":r.get("branch"),"S_feature":r.get("dual_match",{}).get("feature_similarity"),"S_graph":r.get("dual_match",{}).get("graph_similarity"),"joint_candidates":r.get("dual_match",{}).get("joint_candidates",[])},ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")'

NSDI_META_OUTPUT="$OUTPUT_DIR/metadata.json" NSDI_META_ROOT="$ROOT" NSDI_META_CONTRACT="$DATA_CONTRACT" \
NSDI_META_TRAIN_JSONL="$TRAIN_JSONL" NSDI_META_TEST="$EXPANDED_TEST_JSONL" NSDI_META_MODEL="$MODEL_PATH" NSDI_META_CUDA="$CUDA_VISIBLE_DEVICES" \
NSDI_META_TP="$TENSOR_PARALLEL_SIZE" NSDI_META_FREE="$GPU_MIN_FREE_MB" \
NSDI_META_TRAIN="$TRAIN_SIZE" NSDI_META_TEST_SIZE="$EXPECTED_TEST_SIZE" \
"$BASE_PYTHON" -c 'import json,os,platform,datetime,pathlib; p=pathlib.Path(os.environ["NSDI_META_OUTPUT"]); p.write_text(json.dumps({"schema_version":"expanded-remote-run-metadata-v4-dual-sop","data_contract_version":"expanded-expert-clean-v1","finished_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"python":platform.python_version(),"root":os.environ["NSDI_META_ROOT"],"data_contract":os.environ["NSDI_META_CONTRACT"],"train_jsonl":os.environ["NSDI_META_TRAIN_JSONL"],"expanded_test":os.environ["NSDI_META_TEST"],"model_path":os.environ["NSDI_META_MODEL"],"cuda_visible_devices":os.environ["NSDI_META_CUDA"],"tensor_parallel_size":int(os.environ["NSDI_META_TP"]),"gpu_min_free_mb":int(os.environ["NSDI_META_FREE"]),"train_size":int(os.environ["NSDI_META_TRAIN"]),"test_size":int(os.environ["NSDI_META_TEST_SIZE"]),"prompt_and_evaluation":"dual similarity + calibrated pure history + executable SOP + constrained LLM","production_output":"selective","forced_prediction":"observational_only","n8_feedback_update":False},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")'

echo "[expanded-rca] finished: $(timestamp)"
echo "[expanded-rca] result: $OUTPUT_DIR/evaluation_expanded.json"
