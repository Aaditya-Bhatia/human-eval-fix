# HumanEvalFix with vLLM

This runbook explains how to serve `clean-edit-llama-3.2-3b` with vLLM and run the `HumanEvalFix` benchmark against that server.

## Files involved

- vLLM server launcher:
  - `/shared_workspace_mfs/aadi/Projects/EditBench_fork/serve.sh`
- 4-GPU server config:
  - `/shared_workspace_mfs/aadi/Projects/EditBench_fork/configs/clean-edit-llama-3.2-3b-gpu0-3.yaml`
- Benchmark runner:
  - `/shared_workspace_mfs/aadi/Projects/bigcode-evaluation-harness/run_vllm_humanevalfix.py`
- Benchmark virtualenv:
  - `/shared_workspace_mfs/aadi/Projects/bigcode-evaluation-harness/.venv-vllm-bench`

## What this setup does

1. Starts a vLLM OpenAI-compatible server on port `8788`.
2. Exposes the LoRA-backed model as:
   - `clean-edit-llama-3.2-3b`
3. Runs `HumanEvalFix` by sending prompts to `/v1/completions`.
4. Evaluates generated code locally with the BigCode HumanEvalPack scorer.

## Start the vLLM server on GPUs 0-3

```bash
cd /shared_workspace_mfs/aadi/Projects/EditBench_fork
./serve.sh configs/clean-edit-llama-3.2-3b-gpu0-3.yaml
```

Verify the server:

```bash
curl http://127.0.0.1:8788/v1/models
```

You should see model IDs including:

```text
clean-edit-llama-3.2-3b-base
clean-edit-llama-3.2-3b
```

Use `clean-edit-llama-3.2-3b` for the benchmark.

## Run a smoke test

```bash
cd /shared_workspace_mfs/aadi/Projects/bigcode-evaluation-harness
HF_ALLOW_CODE_EVAL=1 .venv-vllm-bench/bin/python run_vllm_humanevalfix.py \
  --base-url http://127.0.0.1:8788 \
  --model clean-edit-llama-3.2-3b \
  --task humanevalfixtests-python \
  --prompt instruct \
  --limit 1 \
  --n-samples 1 \
  --concurrency 1 \
  --max-tokens 1024
```

## Run the full benchmark

```bash
cd /shared_workspace_mfs/aadi/Projects/bigcode-evaluation-harness
HF_ALLOW_CODE_EVAL=1 .venv-vllm-bench/bin/python run_vllm_humanevalfix.py \
  --base-url http://127.0.0.1:8788 \
  --model clean-edit-llama-3.2-3b \
  --task humanevalfixtests-python \
  --prompt instruct \
  --n-samples 1 \
  --concurrency 16 \
  --max-tokens 1024
```

## Run from an EditBench vLLM config

If the vLLM server is already running and you want the benchmark runner to pick up the port, model name, served API name, and benchmark settings from an EditBench YAML config, use:

```bash
cd /shared_workspace_mfs/aadi/Projects/human_eval_fix/bigcode-evaluation-harness
.venv-vllm-bench/bin/python run_benchmark_from_vllm_config.py \
  /shared_workspace_mfs/aadi/Projects/EditBench_fork/configs/qwen3-14b-base-lora-dirty.yaml
```

This wrapper:

- Reads `port`, `model_name`, `served_model_name`, `max_workers`, and `max_tokens` from the YAML config
- Waits for the vLLM server to respond on `/v1/models`
- Runs `run_vllm_humanevalfix.py`
- Renames outputs from the API model name to the benchmark model name when needed
- Sends a Telegram notification through `/shared_workspace_mfs/aadi/Projects/notify_telegram.py` when the run succeeds or fails

## Other task variants

Python docs variant:

```bash
HF_ALLOW_CODE_EVAL=1 .venv-vllm-bench/bin/python run_vllm_humanevalfix.py \
  --base-url http://127.0.0.1:8788 \
  --model clean-edit-llama-3.2-3b \
  --task humanevalfixdocs-python \
  --prompt instruct \
  --n-samples 1 \
  --concurrency 16 \
  --max-tokens 1024
```

The runner also supports:

- `humanevalfixtests-cpp`
- `humanevalfixtests-go`
- `humanevalfixtests-java`
- `humanevalfixtests-js`
- `humanevalfixtests-rust`
- `humanevalfixdocs-cpp`
- `humanevalfixdocs-go`
- `humanevalfixdocs-java`
- `humanevalfixdocs-js`
- `humanevalfixdocs-rust`

## Output files

Outputs are written under:

- `/shared_workspace_mfs/aadi/Projects/bigcode-evaluation-harness/vllm_humanevalfix_runs`

For the full Python test run, the files are:

- generations:
  - `/shared_workspace_mfs/aadi/Projects/bigcode-evaluation-harness/vllm_humanevalfix_runs/humanevalfixtests-python__clean-edit-llama-3.2-3b__instruct__start0__end164.generations.json`
- metrics:
  - `/shared_workspace_mfs/aadi/Projects/bigcode-evaluation-harness/vllm_humanevalfix_runs/humanevalfixtests-python__clean-edit-llama-3.2-3b__instruct__start0__end164.metrics.json`

## Important notes

- `HF_ALLOW_CODE_EVAL=1` is required for evaluation. Without it, the metric refuses to execute generated code.
- The benchmark runner uses `/v1/completions`, not `/v1/chat/completions`.
- The model name must be `clean-edit-llama-3.2-3b`, not the base model path.
- `serve.sh` was adjusted so the LoRA adapter is exposed on the API as `clean-edit-llama-3.2-3b`.
- If another process is already using the same GPUs, the server may fail or contend for memory.

## Current known result

For:

- task: `humanevalfixtests-python`
- prompt: `instruct`
- model: `clean-edit-llama-3.2-3b`
- `n_samples=1`

The recorded score is:

```json
{"pass@1": 0.08536585365853659}
```
