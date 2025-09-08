#!/usr/bin/env bash
set -euo pipefail
# Usage: bash scripts/apply_refactor.sh
# This script is idempotent: it won't overwrite existing files unless explicitly told.

BASE="$(pwd)"
echo "Running apply_refactor in ${BASE}"

# 1. Create package layout if missing
dirs=(
  "fxorcist"
  "fxorcist/data"
  "fxorcist/dashboard"
  "fxorcist/pipeline"
  "fxorcist/ml"
  "fxorcist/models"
  "fxorcist/utils"
  "artifacts"
  ".github/workflows"
)
for d in "${dirs[@]}"; do
  [ -d "$d" ] || mkdir -p "$d"
done

# 2. Touch __init__ files if missing (safe)
init_files=(
  "fxorcist/__init__.py"
  "fxorcist/__main__.py"
  "fxorcist/data/__init__.py"
  "fxorcist/dashboard/__init__.py"
  "fxorcist/pipeline/__init__.py"
  "fxorcist/ml/__init__.py"
  "fxorcist/models/__init__.py"
  "fxorcist/utils/__init__.py"
)
for f in "${init_files[@]}"; do
  [ -f "$f" ] || echo "# created by scripts/apply_refactor.sh" > "$f"
done

# 3. Move existing top-level files to package if (and only if) target missing
mv_if_exists () {
  src="$1"
  dst="$2"
  if [ -f "$src" ] && [ ! -f "$dst" ]; then
    mkdir -p "$(dirname "$dst")"
    git mv "$src" "$dst" || { mv "$src" "$dst"; git add "$dst"; }
    echo "Moved $src -> $dst"
  else
    if [ -f "$dst" ]; then
      echo "SKIP: target exists: $dst"
    elif [ ! -f "$src" ]; then
      echo "SKIP: source not found: $src"
    fi
  fi
}

mv_if_exists "fxorcist_cli.py" "fxorcist/cli.py"
mv_if_exists "prepare_data.py" "fxorcist/data/loader.py"
mv_if_exists "enhanced_training_dashboard.py" "fxorcist/dashboard/app.py"
mv_if_exists "advanced_training_pipeline.py" "fxorcist/models/advanced_training_pipeline.py"
mv_if_exists "run_enhanced_training.py" "fxorcist/models/run_enhanced_training.py"

# 4. Add a backward-compat shim if none exists
if [ ! -f "fxorcist_cli.py" ]; then
  cat > fxorcist_cli.py <<'PY'
from fxorcist.cli import main
if __name__ == '__main__':
    main()
PY
  git add fxorcist_cli.py
  echo "Added shim fxorcist_cli.py"
fi

# 5. Add placeholder files for crucial modules if missing (version, main)
if [ ! -f "fxorcist/__version__.py" ]; then
  echo "__version__ = '0.1.0'" > fxorcist/__version__.py
  git add fxorcist/__version__.py
fi

# 6. Stage and make a single commit if anything changed
if ! git diff --quiet --cached; then
  git commit -m "chore(refactor): scaffold fxorcist package (idempotent apply)"
else
  echo "No staged changes to commit"
fi

echo "apply_refactor finished. Run 'git status' to inspect next steps."