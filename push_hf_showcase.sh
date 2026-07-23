#!/usr/bin/env bash
# Push the stage-evolution showcase to a HuggingFace Space (static HTML).
#
#   1) one-time login:  /group2/ct/weihanx/miniconda3/envs/effecterase/bin/hf auth login
#   2) push:            bash push_hf_showcase.sh [repo_id]
#
# repo_id defaults to av-agent-pipeline-stages under your HF username.
# Created PRIVATE by default; flip to public in the Space settings (or pass
# PUBLIC=1) once you've confirmed you're happy sharing the source clip.
set -euo pipefail

HF=/group2/ct/weihanx/miniconda3/envs/effecterase/bin/hf
REPO_ID="${1:-av-agent-pipeline-stages}"
PRIVATE_FLAG="--private"
[ "${PUBLIC:-0}" = "1" ] && PRIVATE_FLAG=""

cd /group2/ct/weihanx/av_langgraph_pipeline

"$HF" auth whoami >/dev/null 2>&1 || {
  echo "Not logged in. Run: $HF auth login" >&2; exit 1;
}

"$HF" repo create "$REPO_ID" --repo-type space --space_sdk static $PRIVATE_FLAG --exist-ok
"$HF" upload "$REPO_ID" hf_showcase . --repo-type space \
  --commit-message "Stage-by-stage evolution of sample 9NcjRFu6C4I (incl. LTX-2 AV enhancement)"

USER=$("$HF" auth whoami 2>/dev/null | head -1 | awk '{print $NF}')
echo
echo "Pushed. Space URL: https://huggingface.co/spaces/${USER}/${REPO_ID##*/}"
