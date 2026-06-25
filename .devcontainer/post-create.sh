#!/usr/bin/env bash
# .devcontainer/post-create.sh
# Runs ONCE after the container is first created.
# Safe to re-run manually: bash .devcontainer/post-create.sh

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RESET='\033[0m'
step()  { echo -e "\n${CYAN}▶ $*${RESET}"; }
ok()    { echo -e "${GREEN}✔ $*${RESET}"; }
warn()  { echo -e "${YELLOW}⚠ $*${RESET}"; }

# ── 1. Install Poetry (if not already present) ────────────────────────────────
step "Checking Poetry"
if ! command -v poetry &>/dev/null; then
  step "Installing Poetry via official installer"
  curl -sSL https://install.python-poetry.org | python3 -
  export PATH="$HOME/.local/bin:$PATH"
  ok "Poetry installed: $(poetry --version)"
else
  ok "Poetry already present: $(poetry --version)"
fi

# ── 2. Configure Poetry ───────────────────────────────────────────────────────
step "Configuring Poetry"
poetry config virtualenvs.in-project true         # keeps .venv inside workspace
poetry config virtualenvs.prefer-active-python true
ok "Poetry configured (in-project venv)"

# ── 3. Project dependencies via Poetry ───────────────────────────────────────
if [[ -f pyproject.toml ]]; then
  step "Installing project dependencies (poetry install)"
  poetry install --with dev --no-interaction --ansi 2>/dev/null \
    || poetry install --no-interaction --ansi
  ok "Project dependencies installed"
else
  warn "No pyproject.toml found — skipping poetry install"
  warn "Run 'poetry init' to set up your project"
fi

# ── 4. Ensure code-quality tools are available in the venv ───────────────────
# These should live in [tool.poetry.group.dev.dependencies] in pyproject.toml.
# Added here as a safety net if they're missing from the manifest.
step "Ensuring code-quality tools are present"
TOOLS=(black ruff pycodestyle pydocstyle pre-commit)
MISSING_TOOLS=()

for tool in "${TOOLS[@]}"; do
  if ! poetry run "$tool" --version &>/dev/null 2>&1; then
    MISSING_TOOLS+=("$tool")
  fi
done

if [[ ${#MISSING_TOOLS[@]} -gt 0 ]]; then
  warn "Adding missing tools to dev group: ${MISSING_TOOLS[*]}"
  poetry add --group dev "${MISSING_TOOLS[@]}" --no-interaction --ansi
  ok "Missing tools added via poetry"
else
  ok "All code-quality tools present in venv"
fi

# ── 5. Node / JS toolchain ────────────────────────────────────────────────────
if command -v npm &>/dev/null; then
  if [[ -f package.json ]]; then
    step "Installing Node dependencies (package.json detected)"
    npm ci --silent 2>/dev/null || npm install --silent
    ok "Node dependencies installed"
  else
    step "Installing global JS dev tools (prettier, eslint)"
    npm install --save-dev \
      prettier \
      eslint \
      eslint-config-prettier \
      --silent 2>/dev/null || warn "npm install skipped (no package.json)"
  fi
else
  warn "npm not found — skipping JS toolchain"
fi

# ── 6. pre-commit hooks ───────────────────────────────────────────────────────
if [[ -f .pre-commit-config.yaml ]]; then
  step "Installing pre-commit hooks"
  poetry run pre-commit install --install-hooks
  ok "pre-commit hooks installed"
else
  warn ".pre-commit-config.yaml not found — skipping hook installation"
fi

# ── 7. Git safe directory ─────────────────────────────────────────────────────
step "Configuring git safe directory"
git config --global --add safe.directory "$(pwd)" 2>/dev/null || true
ok "git safe directory set"

# ── Done ──────────────────────────────────────────────────────────────────────
echo -e "\n${GREEN}✔ post-create complete — container is ready!${RESET}\n"
