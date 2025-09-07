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

# 1) Add pre-commit config
write_file .pre-commit-config.yaml <<'YAML'
repos:
- repo: https://github.com/psf/black
  rev: 24.4b0
  hooks:
    - id: black
- repo: https://github.com/PyCQA/flake8
  rev: 7.1.0
  hooks:
    - id: flake8
- repo: https://github.com/pre-commit/mirrors-mypy
  rev: v1.9.0
  hooks:
    - id: mypy
      additional_dependencies: [mypy==1.6.1]
YAML

# 2) Add mypy config
write_file mypy.ini <<'MYPY'
[mypy]
python_version = 3.10
ignore_missing_imports = True
MYPY

# 3) Add basic flake8
write_file .flake8 <<'FLAKE'
[flake8]
max-line-length = 120
extend-ignore = E203, W503
FLAKE

# 4) Add updated pyproject.toml
write_file pyproject.toml <<'TOML'
[project]
name = "fxorcist"
version = "0.3.0"
description = "FXorcist — research and backtest framework"
readme = "README.md"
requires-python = ">=3.10"

[project.scripts]
fxorcist = "fxorcist.cli:main"

[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

[tool.black]
line-length = 100
TOML

# 5) requirements + extras
write_file requirements.txt <<'REQ'
streamlit>=1.20.0
plotly>=5.6.0
pandas>=2.0
numpy>=1.24
optuna>=3.0
pyyaml>=6.0
mlflow>=2.0; extra == "mlflow"
dask[distributed]>=2023.9.2
ray[default]>=2.6.0; extra == "ray"
pyarrow
quantstats
rich
pandera
hypothesis
tqdm
aiofiles
fsspec
REQ

# Final commit
if git diff --cached --quiet; then
  echo "No changes staged (files existed, no new writes)."
else
  git commit -m "chore: fxorcist v3 upgrade scaffold (loader, backtest, parallel, optuna, dashboard, ci, tests, pre-commit)"
  echo "Committed changes. Please review with 'git show --name-only HEAD' and run tests."
fi

echo "DONE. Next steps:"
echo "  1) Inspect changes: git diff origin/main..HEAD"
echo "  2) Run tests: pytest -q"
echo "  3) Run pre-commit: pre-commit run --all-files"
echo "  4) Push: git push --set-upstream origin $(git rev-parse --abbrev-ref HEAD)"