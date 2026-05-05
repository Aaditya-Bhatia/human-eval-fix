#!/bin/bash
# migrate.sh — Set up the bigcode-evaluation-harness (HumanEvalFix) on a new server.
#
# This script ONLY handles this project (human_eval_fix + bigcode-evaluation-harness).
# It does NOT touch other repos (SFT, EditBench, Master_VLLM, etc.).
#
# Prerequisites:
#   - Python 3.10+ available
#   - git installed
#   - conda available (for vllm/torch if you want to serve models too)
#
# Usage:
#   cd /path/to/human_eval_fix
#   bash migrate.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="${SCRIPT_DIR}/bigcode-evaluation-harness"
BENCH_VENV="${HARNESS_DIR}/.venv-vllm-bench"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
step() { echo -e "\n${GREEN}==>${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }

# ============================================================================
# 1. Clone the evaluation harness (our fork)
# ============================================================================
if [ ! -d "${HARNESS_DIR}/.git" ]; then
    step "Cloning bigcode-evaluation-harness fork..."
    git clone https://github.com/Aaditya-Bhatia/bigcode-evaluation-harness.git "${HARNESS_DIR}"
else
    step "bigcode-evaluation-harness already present — ensuring fork remote exists..."
    cd "${HARNESS_DIR}"
    git remote set-url origin https://github.com/Aaditya-Bhatia/bigcode-evaluation-harness.git 2>/dev/null || true
    git remote add upstream https://github.com/bigcode-project/bigcode-evaluation-harness.git 2>/dev/null || true
    cd "${SCRIPT_DIR}"
fi

# ============================================================================
# 2. Create the benchmark virtualenv
# ============================================================================
if [ ! -d "${BENCH_VENV}" ]; then
    step "Creating benchmark virtualenv (.venv-vllm-bench)..."
    python3 -m venv "${BENCH_VENV}"
else
    step "Benchmark venv already exists — skipping creation."
fi

step "Installing harness requirements..."
"${BENCH_VENV}/bin/pip" install --upgrade pip
"${BENCH_VENV}/bin/pip" install -r "${HARNESS_DIR}/requirements.txt"

step "Installing additional runner dependencies..."
"${BENCH_VENV}/bin/pip" install openai requests tqdm pyyaml

# ============================================================================
# 3. Download the HumanEvalPack dataset (if not cached)
# ============================================================================
step "Pre-downloading humanevalpack dataset..."
"${BENCH_VENV}/bin/python" -c "
from datasets import load_dataset
print('  Downloading bigcode/humanevalpack...')
load_dataset('bigcode/humanevalpack', 'python', split='test')
print('  Done.')
" || warn "Dataset download failed — will be fetched on first run"

# ============================================================================
# Done
# ============================================================================
echo ""
echo -e "${GREEN}========================================"
echo "  Setup complete!"
echo -e "========================================${NC}"
echo ""
echo "To run HumanEvalFix against a vLLM server:"
echo ""
echo "  cd ${HARNESS_DIR}"
echo "  HF_ALLOW_CODE_EVAL=1 .venv-vllm-bench/bin/python run_vllm_humanevalfix.py \\"
echo "    --base-url http://127.0.0.1:8788 \\"
echo "    --model <MODEL_NAME> \\"
echo "    --task humanevalfixtests-python \\"
echo "    --prompt instruct \\"
echo "    --n-samples 1 \\"
echo "    --concurrency 16 \\"
echo "    --max-tokens 1024"
echo ""
echo "Or use the config-based runner:"
echo ""
echo "  .venv-vllm-bench/bin/python run_benchmark_from_vllm_config.py <path-to-vllm-config.yaml>"
