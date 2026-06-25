#!/usr/bin/env bash
# .devcontainer/post-start.sh
# Runs every time the container starts (after rebuild AND resume).
# Keep this fast — it runs on every VS Code attach.

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RESET='\033[0m'
step()  { echo -e "\n${CYAN}▶ $*${RESET}"; }
ok()    { echo -e "${GREEN}✔ $*${RESET}"; }
warn()  { echo -e "${YELLOW}⚠ $*${RESET}"; }

# ── Ensure Poetry is on PATH ──────────────────────────────────────────────────
export PATH="$HOME/.local/bin:$PATH"

# ── 1. Verify core tools are present ─────────────────────────────────────────
step "Checking tool availability"

if ! command -v poetry &>/dev/null; then
  warn "Poetry not found — re-running post-create to fix..."
  bash "$(dirname "$0")/post-create.sh"
  exit 0
fi

MISSING=()
for tool in black ruff pycodestyle pydocstyle pre-commit; do
  if ! poetry run "$tool" --version &>/dev/null 2>&1; then
    MISSING+=("$tool")
  fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
  warn "Missing tools in venv: ${MISSING[*]} — re-running post-create to fix..."
  bash "$(dirname "$0")/post-create.sh"
else
  ok "All tools present in Poetry venv"
fi

# ── 2. Sync dependencies (fast — only acts if poetry.lock changed) ────────────
if [[ -f pyproject.toml ]]; then
  step "Syncing dependencies (poetry install --sync)"
  poetry install --with dev --sync --no-interaction --quiet 2>/dev/null \
    || poetry install --no-interaction --quiet
  ok "Dependencies up to date"
fi

# ── 3. Ensure pre-commit hooks are installed ──────────────────────────────────
if [[ -f .pre-commit-config.yaml ]]; then
  if [[ ! -f .git/hooks/pre-commit ]]; then
    step "Re-installing pre-commit hooks (missing from .git/hooks)"
    poetry run pre-commit install --install-hooks --quiet
    ok "pre-commit hooks installed"
  else
    ok "pre-commit hooks already installed"
  fi
fi

# ── 4. Node dependencies (only if lockfile is newer than node_modules) ────────
if [[ -f package.json ]] && command -v npm &>/dev/null; then
  if [[ ! -d node_modules ]] || \
     [[ package-lock.json -nt node_modules/.package-lock.json 2>/dev/null ]]; then
    step "Syncing Node dependencies"
    npm ci --silent 2>/dev/null || npm install --silent
    ok "Node dependencies up to date"
  else
    ok "Node dependencies already up to date"
  fi
fi

# ── 5. Environment summary ────────────────────────────────────────────────────
step "Environment summary"
echo "  Python  : $(poetry run python --version 2>&1)"
echo "  Poetry  : $(poetry --version 2>&1)"
echo "  black   : $(poetry run black --version 2>&1 | head -1)"
echo "  ruff    : $(poetry run ruff --version 2>&1)"
echo "  Node    : $(node --version 2>/dev/null || echo 'not available')"
echo "  npm     : $(npm --version 2>/dev/null || echo 'not available')"
echo "  Workspace: $(pwd)"

# ── Done ──────────────────────────────────────────────────────────────────────
echo -e "\n${GREEN}✔ Container ready — happy coding!${RESET}\n"
