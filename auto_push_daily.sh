#!/usr/bin/env bash
# Commit and push local changes once per scheduled run.

set -euo pipefail

repo_dir="/mnt/lv/home/ztw/SIGKDD2027-BatteryTwin-Code-and-Dataset"
log_file="/mnt/lv/home/ztw/.sigkdd2027_auto_push.log"

exec >>"$log_file" 2>&1
printf '\n[%s] Daily sync started\n' "$(date '+%F %T %Z')"

cd "$repo_dir"

if [[ -z "$(git status --porcelain)" ]]; then
  echo "No local changes; nothing to push."
  exit 0
fi

git add --all
git commit -m "chore: automated daily backup $(date '+%F')"
git push origin main

echo "Changes committed and pushed successfully."
