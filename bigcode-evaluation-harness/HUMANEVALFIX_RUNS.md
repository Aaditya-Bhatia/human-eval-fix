# HumanEvalFix Benchmark Runs

## Overview

The supported entrypoints are:

- `run_from_config.sh` — **the Master_VLLM entrypoint.** Activates the benchmark venv and dispatches to `run_benchmark_from_vllm_config.py` with `HF_ALLOW_CODE_EVAL=1` set. This is the script the dual-slot scheduler in `/shared_workspace_mfs/aadi/Projects/Master_VLLM/master_vllm.py` invokes for each model.
- `run_benchmark_from_vllm_config.py` — Python implementation: reads the Master_VLLM runtime YAML, waits on `/v1/models`, runs HumanEvalFix, renames outputs back to the config's `model_name`, and sends Telegram notifications.
- `run_vllm_humanevalfix.py` — low-level runner for direct execution against an already-running vLLM OpenAI-compatible server.

Legacy shell runners were removed in favor of the config-driven wrapper.

## Recommended workflow (Master_VLLM managed)

Master_VLLM starts a vLLM server using its scheduled runtime YAML, then spawns:

```bash
/shared_workspace_mfs/aadi/Projects/human_eval_fix/bigcode-evaluation-harness/run_from_config.sh \
  /shared_workspace_mfs/aadi/Projects/Master_VLLM/runtime/slot0-humanevalfix-<model>.yaml
```

The Master_VLLM runtime YAML is the same shape as an EditBench config (`port`, `model_name`, optional `served_model_name`, `max_workers`, `max_tokens`), and `run_from_config.sh` is a thin activator for the venv used by the benchmark runner.

## Manual workflow

You can still call the underlying runner directly with any EditBench-style YAML:

```bash
cd /shared_workspace_mfs/aadi/Projects/human_eval_fix/bigcode-evaluation-harness
.venv-vllm-bench/bin/python run_benchmark_from_vllm_config.py \
  /shared_workspace_mfs/aadi/Projects/EditBench_fork/configs/qwen3-14b-base-lora-dirty.yaml
```

The wrapper:

1. Reads `port`, `model_name`, optional `served_model_name`, `max_workers`, and `max_tokens` from the YAML config
2. Waits for the configured vLLM server to respond on `/v1/models`
3. Runs the HumanEvalFix benchmark
4. Renames outputs back to `model_name` if the API serves a different model ID
5. Sends a Telegram notification on success or failure through `/shared_workspace_mfs/aadi/Projects/notify_telegram.py`

## Direct runner

If you already know the server URL and exposed model ID, you can still run the lower-level benchmark directly:

```bash
cd /shared_workspace_mfs/aadi/Projects/human_eval_fix/bigcode-evaluation-harness
HF_ALLOW_CODE_EVAL=1 .venv-vllm-bench/bin/python run_vllm_humanevalfix.py \
  --base-url http://127.0.0.1:9363 \
  --model editbench_adapter \
  --task humanevalfixtests-python \
  --prompt instruct \
  --n-samples 1 \
  --concurrency 32 \
  --max-tokens 12288
```

## Output

Outputs are written under `vllm_humanevalfix_runs/`:

- `*.generations.json`: model completions
- `*.metrics.json`: benchmark results such as `pass@1`
- `*.errors.json`: request failures, if any

## Notes

- `HF_ALLOW_CODE_EVAL=1` is still required for evaluation.
- The benchmark runner uses `/v1/completions`, not `/v1/chat/completions`.
- For LoRA-backed servers, the benchmark wrapper handles the case where `served_model_name` differs from `model_name`.
