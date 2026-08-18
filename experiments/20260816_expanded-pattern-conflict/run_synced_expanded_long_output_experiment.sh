#!/usr/bin/env bash
set -euo pipefail

# Quick controlled rerun after the 512-token experiment truncated 339/340
# structured responses.  All data, matching, routing, SOP and model settings
# remain unchanged; only the context/output budgets use explicit larger values.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export NSDI_RCA_MAX_MODEL_LEN="${NSDI_RCA_MAX_MODEL_LEN:-16384}"
export NSDI_RCA_MAX_NEW_TOKENS="${NSDI_RCA_MAX_NEW_TOKENS:-2048}"

echo "[expanded-long-output] max_model_len=$NSDI_RCA_MAX_MODEL_LEN"
echo "[expanded-long-output] max_new_tokens=$NSDI_RCA_MAX_NEW_TOKENS"
echo "[expanded-long-output] all other experiment settings are inherited unchanged"

exec bash "$SCRIPT_DIR/run_synced_expanded_experiment.sh"
