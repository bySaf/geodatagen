#!/usr/bin/env bash
# Optional accelerator extras; CPU-only setup remains scripts/setup.sh.
set -euo pipefail
REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$REPO_ROOT"
if [[ ! -x .venv/bin/python ]]; then echo 'Run bash scripts/setup.sh first.' >&2; exit 1; fi
case "${1:-}" in
  metal)
    if [[ "$(uname -s)" != Darwin ]]; then echo 'Metal requires macOS.' >&2; exit 1; fi
    .venv/bin/python -m pip install 'gprMax[metal]==4.0.1'
    ;;
  cuda)
    if ! command -v nvcc >/dev/null 2>&1; then echo 'Install NVIDIA CUDA toolkit and driver first (nvcc required).' >&2; exit 1; fi
    .venv/bin/python -m pip install 'gprMax[cuda]==4.0.1'
    ;;
  *) echo 'Usage: bash scripts/setup-gpu.sh metal|cuda' >&2; exit 2 ;;
esac
.venv/bin/python -m pip check
.venv/bin/python geodatagen.py backends
