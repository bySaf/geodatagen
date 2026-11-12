#!/usr/bin/env bash
# Install only into this checkout's .venv. No sudo or global Python changes.
set -euo pipefail
REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$REPO_ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/matplotlib}"
export PIP_CACHE_DIR="$REPO_ROOT/.cache/pip"

usage() {
  cat <<'EOF'
GeoDataGen — setup for macOS / Linux / Windows WSL2

  bash scripts/setup.sh          create .venv and install the pinned CPU runtime
  bash scripts/setup.sh --check  inspect the existing environment without changes
  bash scripts/setup.sh --help   show this help

Requires Python 3.12 (recommended) or 3.13. To choose an interpreter:
  GEODATAGEN_PYTHON=/path/to/python3.12 bash scripts/setup.sh

gprMax is installed from a binary wheel. No compiler is needed on supported
macOS arm64/x86_64 and Linux x86_64. ParaView is a separate optional application.
EOF
}

check_python_version() {
  "$1" - <<'PY'
import sys
if sys.version_info[:2] not in ((3, 12), (3, 13)):
    raise SystemExit(
        f'GeoDataGen requires Python 3.12 or 3.13; selected {sys.version.split()[0]}.\n'
        f'Interpreter: {sys.executable}\n'
        'Create a new virtual environment using Python 3.12, then select its bin/python in PyCharm.\n'
        'Installing or downgrading contourpy does not fix an incompatible interpreter.\n'
        'The existing environment has not been changed.')
PY
}

check_environment() {
  if [[ ! -x .venv/bin/python ]]; then
    echo 'Environment not found. Run: bash scripts/setup.sh' >&2
    return 1
  fi
  check_python_version .venv/bin/python
  .venv/bin/python - <<'PY'
import importlib.metadata as m
import sys
import gprMax, h5py, matplotlib, numpy, scipy
expected = '4.0.1'
actual = m.version('gprMax')
if actual != expected:
    raise SystemExit(f'Expected gprMax {expected}; found {actual}.')
print(f'Python {sys.version.split()[0]} · gprMax {actual} · CPU runtime ready')
print(f'Interpreter: {sys.executable}')
PY
  .venv/bin/python -m pip check
}

case "${1:-}" in
  --help|-h) usage; exit 0 ;;
  --check) check_environment; exit 0 ;;
  '') ;;
  *) usage >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then usage >&2; exit 2; fi

if [[ -e .venv && ! -x .venv/bin/python ]]; then
  echo '.venv already exists but is not a usable Unix environment. Move it aside before setup.' >&2
  exit 1
fi
if [[ -x .venv/bin/python ]]; then
  PYTHON_BIN=.venv/bin/python
else
  PYTHON_BIN="${GEODATAGEN_PYTHON:-}"
  if [[ -z "$PYTHON_BIN" ]]; then
    for candidate in python3.12 python3.13 python3; do
      if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; raise SystemExit(sys.version_info[:2] not in [(3,12),(3,13)])' 2>/dev/null; then
        PYTHON_BIN="$candidate"
        break
      fi
    done
  fi
  if [[ -z "$PYTHON_BIN" ]]; then
    echo 'Install Python 3.12 first: https://www.python.org/downloads/' >&2
    echo 'On Ubuntu/WSL also install the matching python3.12-venv package.' >&2
    exit 1
  fi
fi
check_python_version "$PYTHON_BIN"
if [[ ! -x .venv/bin/python ]]; then "$PYTHON_BIN" -m venv .venv; fi

echo 'Installing pinned CPU dependencies in .venv…'
.venv/bin/python -m pip install --upgrade pip
if ! .venv/bin/python -m pip install --only-binary=gprMax -r requirements-lock.txt; then
  echo 'Installation failed. Check the network and Python/platform compatibility.' >&2
  echo 'Source builds are not attempted automatically. Upstream instructions:' >&2
  echo 'https://github.com/gprMax/gprMax#installation' >&2
  exit 1
fi
check_environment
cat <<'EOF'

Ready. Next:
  source .venv/bin/activate
  python geodatagen.py docs
  python geodatagen.py init first-test --scene examples/scenes/smoke.json --trajectory examples/trajectories/smoke.json
  python geodatagen.py run first-test --limit 1

Experiment files stay in experiments/first-test/.
EOF
