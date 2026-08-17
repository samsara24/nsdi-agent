#!/usr/bin/env bash
set -Eeuo pipefail

# Git-synchronized wrapper for the expanded expert-clean experiment. It refuses
# a dirty tree so the pulled input commit and uploaded output remain auditable.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NSDI_RCA_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
REMOTE="${NSDI_RCA_GIT_REMOTE:-origin}"
BRANCH="${NSDI_RCA_GIT_BRANCH:-codex/expanded-expert-clean-v1}"
RUNNER="$ROOT/experiments/20260816_expanded-pattern-conflict/run_expanded_remote_experiment.sh"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${NSDI_RCA_OUTPUT_DIR:-$ROOT/artifacts/expanded_expert_clean_deepseek32b_$TIMESTAMP}"
SYNC_REEXECED="${NSDI_RCA_SYNC_REEXECED:-0}"
GIT_AUTHOR_NAME="${NSDI_RCA_GIT_AUTHOR_NAME:-RCA Experiment Runner}"
GIT_AUTHOR_EMAIL="${NSDI_RCA_GIT_AUTHOR_EMAIL:-rca-experiment@users.noreply.github.com}"

timestamp() { date --iso-8601=seconds 2>/dev/null || date; }
die() { echo "[expanded-sync] ERROR: $*" >&2; exit 2; }

command -v git >/dev/null 2>&1 || die "git is required"
[[ -d "$ROOT/.git" ]] || die "not a Git checkout: $ROOT"
[[ -x "$RUNNER" ]] || die "experiment runner is missing or not executable: $RUNNER"
cd "$ROOT"

git remote get-url "$REMOTE" >/dev/null 2>&1 || die "Git remote does not exist: $REMOTE"

if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "[expanded-sync] refusing to pull over a dirty worktree:" >&2
  git status --short >&2
  echo "[expanded-sync] commit, upload, or move these files before retrying." >&2
  exit 2
fi

echo "[expanded-sync] fetch $REMOTE/$BRANCH at $(timestamp)"
git fetch "$REMOTE" "$BRANCH"
git show-ref --verify --quiet "refs/remotes/$REMOTE/$BRANCH" \
  || die "remote branch does not exist: $REMOTE/$BRANCH"

CURRENT_BRANCH="$(git branch --show-current)"
if [[ "$CURRENT_BRANCH" != "$BRANCH" ]]; then
  if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
    git switch "$BRANCH"
  else
    git switch --track -c "$BRANCH" "$REMOTE/$BRANCH"
  fi
fi

BEFORE_PULL="$(git rev-parse HEAD)"
git pull --ff-only "$REMOTE" "$BRANCH"
AFTER_PULL="$(git rev-parse HEAD)"
echo "[expanded-sync] reproducible input commit: $AFTER_PULL"

# Restart once when the pull changed this wrapper, ensuring the new version is
# the one that controls the experiment.
if [[ "$BEFORE_PULL" != "$AFTER_PULL" && "$SYNC_REEXECED" != "1" ]]; then
  echo "[expanded-sync] code changed during pull; restarting with the new wrapper"
  export NSDI_RCA_SYNC_REEXECED=1
  exec bash "$ROOT/experiments/20260816_expanded-pattern-conflict/run_synced_expanded_experiment.sh"
fi

case "$OUTPUT_DIR" in
  "$ROOT"/*) ;;
  *) die "NSDI_RCA_OUTPUT_DIR must be inside $ROOT so results can be uploaded" ;;
esac
OUTPUT_REL="${OUTPUT_DIR#"$ROOT/"}"
[[ ! -e "$OUTPUT_DIR" ]] || die "refusing to overwrite output: $OUTPUT_DIR"

export NSDI_RCA_ROOT="$ROOT"
export NSDI_RCA_OUTPUT_DIR="$OUTPUT_DIR"

echo "[expanded-sync] run output: $OUTPUT_REL"
TEMP_RUN_LOG="$(mktemp "${TMPDIR:-/tmp}/expanded-sync-run.XXXXXX")"
cleanup_temp_log() { rm -f "$TEMP_RUN_LOG"; }
trap cleanup_temp_log EXIT
set +e
bash "$RUNNER" 2>&1 | tee "$TEMP_RUN_LOG"
RUN_STATUS="${PIPESTATUS[0]}"
set -e

# The lower-level runner creates this directory after its static preflight. If
# that preflight fails earlier, create it here so the console error is still
# versioned and uploaded.
mkdir -p "$OUTPUT_DIR"
cp "$TEMP_RUN_LOG" "$OUTPUT_DIR/sync_runner_console.log"
{
  printf 'schema_version=expanded-git-sync-v1\n'
  printf 'git_remote=%s\n' "$REMOTE"
  printf 'git_branch=%s\n' "$BRANCH"
  printf 'input_commit=%s\n' "$AFTER_PULL"
  printf 'experiment_exit_status=%s\n' "$RUN_STATUS"
  printf 'upload_started_at=%s\n' "$(timestamp)"
} > "$OUTPUT_DIR/git_sync.txt"

# Only the unique result directory may enter the automatic commit.
OUTSIDE_CHANGES="$(git status --porcelain -- . ":(exclude)$OUTPUT_REL")"
if [[ -n "$OUTSIDE_CHANGES" ]]; then
  echo "[expanded-sync] refusing automatic commit; files outside $OUTPUT_REL changed:" >&2
  printf '%s\n' "$OUTSIDE_CHANGES" >&2
  exit 5
fi

git add -- "$OUTPUT_REL"
git diff --cached --quiet && die "no experiment outputs were staged"

RUN_ID="$(basename "$OUTPUT_DIR")"
if [[ "$RUN_STATUS" == "0" ]]; then
  RESULT="success"
else
  RESULT="failed-exit-$RUN_STATUS"
fi
git -c user.name="$GIT_AUTHOR_NAME" -c user.email="$GIT_AUTHOR_EMAIL" \
  commit -m "Experiment $RUN_ID: $RESULT"

# A long GPU run may overlap another push. Rebase the unique output commit onto
# the latest branch; a real conflict stops visibly and is never force-pushed.
git fetch "$REMOTE" "$BRANCH"
git rebase "$REMOTE/$BRANCH"
git push "$REMOTE" "HEAD:$BRANCH"

PUBLISHED_COMMIT="$(git rev-parse HEAD)"
echo "[expanded-sync] uploaded $OUTPUT_REL"
echo "[expanded-sync] published commit: $PUBLISHED_COMMIT"
echo "[expanded-sync] experiment exit status: $RUN_STATUS"
exit "$RUN_STATUS"
