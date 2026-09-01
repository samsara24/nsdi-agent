#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

[[ -z "$(git status --porcelain)" ]] || { echo "working tree must be clean before formal experiment" >&2; exit 2; }
git fetch origin main
if git show-ref --verify --quiet refs/heads/main; then
  git switch main
else
  git switch --track -c main origin/main
fi
git pull --ff-only origin main
[[ -z "$(git status --porcelain)" ]] || { echo "working tree changed during sync" >&2; exit 2; }

start_revision="$(git rev-parse HEAD)"
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_dir="${OUTPUT_DIR:-$repo_root/artifacts/filtered_rule_temporal_${run_stamp}}"
export OUTPUT_DIR="$output_dir"

default_test_python="/home/chenziang/miniconda3/bin/python3"
if [[ -x "$default_test_python" ]]; then
  test_python="${TEST_PYTHON:-$default_test_python}"
else
  test_python="${TEST_PYTHON:-${PYTHON_BIN:-python}}"
fi
"$test_python" -m pytest -q
scripts/run_filtered_rule_temporal_gpu_experiment.sh

git_status="$(git status --porcelain)"
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  changed_path="${line:3}"
  [[ "$changed_path" == "${output_dir#$repo_root/}" || "$changed_path" == "${output_dir#$repo_root/}/"* ]] || {
    echo "unexpected change after experiment: $changed_path" >&2
    exit 5
  }
done <<< "$git_status"

git add "$output_dir"
git commit -m "experiment: filtered-rule temporal dual-test $run_stamp"
git fetch origin main
git rebase origin/main
git push origin HEAD:main
echo "start_revision=$start_revision"
echo "result_revision=$(git rev-parse HEAD)"
echo "results=$output_dir"
