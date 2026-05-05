#!/usr/bin/env bash
# Reads an EditBench-style YAML config and runs the HumanEvalFix benchmark
# against an already-running vLLM server. Intended as the Master_VLLM runner
# entrypoint for HumanEvalFix, mirroring CanItEdit's run_from_config.sh.
#
# Usage: ./run_from_config.sh <config.yaml> [--task NAME] [--prompt NAME]
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv-vllm-bench/bin/python"
RUNNER="$SCRIPT_DIR/run_benchmark_from_vllm_config.py"

if [[ $# -lt 1 ]]; then
    echo "Usage: $(basename "$0") <config.yaml> [extra args forwarded to run_benchmark_from_vllm_config.py]"
    exit 1
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "Error: venv python not found at $VENV_PYTHON"
    echo "Create it per /shared_workspace_mfs/aadi/Projects/human_eval_fix/bigcode-evaluation-harness/HUMANEVALFIX_RUNS.md"
    exit 1
fi

if [[ ! -f "$RUNNER" ]]; then
    echo "Error: runner not found at $RUNNER"
    exit 1
fi

export HF_ALLOW_CODE_EVAL=1
exec "$VENV_PYTHON" "$RUNNER" "$@"
