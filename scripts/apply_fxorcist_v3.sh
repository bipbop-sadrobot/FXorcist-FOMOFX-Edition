#!/usr/bin/env bash
# scripts/apply_fxorcist_v3.sh
# Idempotent apply script for FXorcist v3 upgrade.
# Usage: bash scripts/apply_fxorcist_v3.sh [--force]
set -euo pipefail

FORCE=0
if [ "${1:-}" = "--force" ]; then
  FORCE=1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "Applying FXorcist v3 upgrade in $REPO_ROOT (force=${FORCE})"

# Ensure we're not on main
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ "$CURRENT_BRANCH" = "main" ] || [ "$CURRENT_BRANCH" = "master" ]; then
  echo "ERROR: Please run this script on a feature branch (not main/master). Create one now:"
  echo "  git checkout -b refactor/v3-upgrade"
  exit 1
fi

# Create directories
mkdir -p fxorcist/data fxorcist/pipeline fxorcist/ml fxorcist/dashboard fxorcist/utils scripts tests .github/workflows

# Helper to write file only if missing or force=1
write_file() {
  local path="$1"; shift
  local tmp=$(mktemp)
  cat > "$tmp"
  if [ -f "$path" ] && [ "$FORCE" -eq 0 ]; then
    echo "SKIP (exists): $path"
    rm "$tmp"
  else
    mkdir -p "$(dirname "$path")"
    mv "$tmp" "$path"
    git add "$path"
    echo "WROTE: $path"
  fi
}