# Server Migration Guide

This documents the full setup needed to replicate the current environment on a new GPU server.

## Prerequisites

- New server with GPUs (4-8x recommended)
- Shared filesystem mounted at `/shared_workspace_mfs/aadi` (if using NFS/similar), OR you'll need to copy the data
- CUDA drivers installed at the OS level
- `git`, `curl`, `wget` available

## What's Already Portable (on shared filesystem)

If `shared_workspace_mfs` is an NFS mount shared across servers, these are already available:

- **Miniconda**: `/shared_workspace_mfs/aadi/miniconda3`
- **Conda envs** (SFT_env, canitedit, codeeditorbench, etc.)
- **Node.js (fnm)**: `/shared_workspace_mfs/aadi/.fnm`
- **All project code**: `/shared_workspace_mfs/aadi/Projects/`
- **Trained LoRA adapters**: `/shared_workspace_mfs/aadi/Projects/SFT/SFT_runs/model/`
- **HuggingFace cache**: `/shared_workspace_mfs/aadi/Models/`
- **Secrets**: `/shared_workspace_mfs/aadi/.server-setup/secrets.env`
- **Shell config**: `/shared_workspace_mfs/aadi/common.bashrc`

**If NFS is shared, you only need to source `common.bashrc` on the new server and you're done.**

## If Starting Fresh (no shared filesystem)

Run `migrate.sh` (see below) which handles everything automatically.

---

## Architecture Overview

```
/shared_workspace_mfs/aadi/
├── common.bashrc              # Shell config (source this in ~/.bashrc)
├── miniconda3/                # Miniconda install + all conda envs
│   └── envs/
│       ├── SFT_env/           # Main env: vLLM, torch, transformers, peft, trl, LLaMA-Factory
│       ├── canitedit/
│       ├── codeeditorbench/
│       ├── llms_env/
│       └── repost/
├── .fnm/                      # Node.js via fnm
├── .npm-global/               # npm global packages (Claude CLI)
├── .bun/                      # Bun runtime
├── .server-setup/
│   └── secrets.env            # API keys (HF_TOKEN, WANDB, OPENAI, ANTHROPIC, etc.)
├── .claude-config/            # Claude Code config
├── .claude-data/              # Claude Code data
├── Models/                    # HuggingFace model cache (HF_HOME)
│   └── hub/                   # Downloaded model snapshots
├── Projects/
│   ├── SFT/                   # Training pipeline (LLaMA-Factory based)
│   │   ├── setup.sh           # One-shot env bootstrap
│   │   ├── pipeline.sh        # Training launcher
│   │   ├── SFT_runs/model/    # Trained LoRA adapters
│   │   └── lib/LLaMA-Factory/ # Patched LLaMA-Factory
│   ├── EditBench_fork_contaminated/  # vLLM server + EditBench eval
│   │   ├── serve.sh           # Unified vLLM launcher (reads YAML configs)
│   │   └── configs/           # Per-model YAML configs
│   ├── Master_VLLM/           # Orchestrator: auto-schedules all benchmarks
│   │   └── master_vllm.py     # build-manifest / run
│   ├── human_eval_fix/        # HumanEvalFix benchmark
│   │   └── bigcode-evaluation-harness/
│   │       ├── run_vllm_humanevalfix.py
│   │       ├── run_benchmark_from_vllm_config.py
│   │       └── .venv-vllm-bench/   # Separate venv for benchmark client
│   ├── CanItEdit/
│   ├── CodeEditorBench/
│   └── notify_telegram.py     # Telegram notifications
└── .ssh/                      # SSH keys (symlinked into ~/.ssh)
```

## Key Conda Environment: `SFT_env`

This is the workhorse env used for both training AND serving:

| Package | Version |
|---------|---------|
| torch | 2.10.0 (or 2.8.0 in requirements.txt — use latest compatible) |
| vllm | 0.17.1 |
| transformers | 4.57.6 |
| peft | 0.18.1 |
| trl | 0.24.0 |
| accelerate | 1.10.1 |
| flash_attn | 2.8.3 |
| deepspeed | 0.16.9 |
| CUDA (conda) | 12.8.1 |

## Base Models Location

Base models are at `/home/original_models/` (server-local, NOT on shared fs).
These need to be downloaded per-server:

```
models--meta-llama--Llama-3.2-3B
models--Qwen--Qwen2.5-Coder-3B
models--Qwen--Qwen2.5-Coder-3B-Instruct
models--Qwen--Qwen2.5-Coder-7B
models--Qwen--Qwen2.5-Coder-14B
models--Qwen--Qwen2.5-Coder-14B-Instruct
models--codellama--CodeLlama-7b-hf
models--codellama--CodeLlama-13b-hf
models--deepseek-ai--deepseek-coder-6.7b-base
(+ others in /home/original_models/)
```

## Quick Start on New Server

```bash
# 1. Source the shared bashrc
echo 'source /shared_workspace_mfs/aadi/common.bashrc' >> ~/.bashrc
source /shared_workspace_mfs/aadi/common.bashrc

# 2. If base models aren't at /home/original_models, download them:
conda activate SFT_env
python -c "
from huggingface_hub import snapshot_download
models = [
    'meta-llama/Llama-3.2-3B',
    'Qwen/Qwen2.5-Coder-3B',
    'Qwen/Qwen2.5-Coder-3B-Instruct',
    'Qwen/Qwen2.5-Coder-7B',
    'Qwen/Qwen2.5-Coder-14B',
    'Qwen/Qwen2.5-Coder-14B-Instruct',
    'codellama/CodeLlama-7b-hf',
    'codellama/CodeLlama-13b-hf',
    'deepseek-ai/deepseek-coder-6.7b-base',
]
for m in models:
    print(f'Downloading {m}...')
    snapshot_download(m, cache_dir='/home/original_models')
"

# 3. Run benchmarks (Master_VLLM handles everything)
cd /shared_workspace_mfs/aadi/Projects/Master_VLLM
python3 master_vllm.py build-manifest
python3 master_vllm.py run --slots 0,1
```

## If Shared Filesystem Is NOT Available

Use `migrate.sh` below to bootstrap from scratch.

---

## Secrets Required (fill in secrets.env)

```bash
# /shared_workspace_mfs/aadi/.server-setup/secrets.env
export HF_TOKEN=<your-hf-token>
export WANDB_API_KEY=<your-wandb-key>
export OPENAI_API_KEY=<your-openai-key>
export ANTHROPIC_AUTH_TOKEN=<your-anthropic-key>
# (see current secrets.env for full list)
```

## Running Things

### Training (SFT with LoRA)
```bash
conda activate SFT_env
cd /shared_workspace_mfs/aadi/Projects/SFT
./pipeline.sh --config SFT_runs/conf/your_config.yaml
```

### Serving a model
```bash
conda activate SFT_env
cd /shared_workspace_mfs/aadi/Projects/EditBench_fork_contaminated
./serve.sh configs/llama-3.2-3b-lora-clean.yaml
```

### Running HumanEvalFix against a running vLLM
```bash
cd /shared_workspace_mfs/aadi/Projects/human_eval_fix/bigcode-evaluation-harness
.venv-vllm-bench/bin/python run_benchmark_from_vllm_config.py \
  /shared_workspace_mfs/aadi/Projects/EditBench_fork_contaminated/configs/llama-3.2-3b-lora-clean.yaml
```

### Full automated pipeline (serve + eval all pending models)
```bash
cd /shared_workspace_mfs/aadi/Projects/Master_VLLM
python3 master_vllm.py run --slots 0,1
```

## Config Adjustments for New Server

If the new server has different GPU count or layout, edit the YAML configs in `EditBench_fork_contaminated/configs/` to update:
- `gpus:` field (e.g., "0,1,2,3" → "0,1")
- `tensor_parallel_size` / `data_parallel_size`
- `port:` if there are conflicts

If base models are at a different path than `/home/original_models/`, update `model_path` in the YAML configs.
