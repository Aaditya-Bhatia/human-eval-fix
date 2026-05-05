# HumanEvalFix Benchmark Results

**Task:** `humanevalfixtests-python` (164 problems)  
**Prompt:** `instruct` | **n_samples:** 1 | **max_tokens:** 12288 | **temperature:** 0.2 | **top_p:** 0.95

## Naming Convention

Run names in this document use the following reporting labels:

| Reported Run Name | Meaning | Sample Size |
|---|---|---|
| Baseline | Base model without filtering fine-tune | N/A |
| All Unclean Sample | Formerly `unclean74k` | 74K |
| Static Quality Filtered | Formerly `dirty` | 50K |
| LLM-Quality Filtered | Formerly `clean` | Original run sample |

## Model Comparison

| Model | Baseline | All Unclean Sample (74K) | Static Quality Filtered (50K) | LLM-Quality Filtered (21.7K) |
|---|---:|---:|---:|---:|
| Qwen3-14B (Base) | 32.3% | 45.7% | 47.6% | **62.8%** |
| Qwen2.5-Coder-7B | 20.7% | 36.0% | 33.5% | **48.2%** |
| Qwen2.5-Coder-3B | 19.5% | 24.4% | 23.8% | **31.1%** |
| DeepSeek-Coder-6.7B | **45.7%** | 29.9% | 26.8% | 37.8% |
| LLaMA-3.2-3B | 1.2% | 3.7% | 5.5% | **8.5%** |

## Ranked Results

| Rank | Model | Run Type | Sample Size | pass@1 |
|---|---|---|---|---:|
| 1 | Qwen3-14B (Base) | LLM-Quality Filtered | Original run sample | **62.8%** |
| 2 | Qwen2.5-Coder-7B | LLM-Quality Filtered | Original run sample | **48.2%** |
| 3 | Qwen3-14B (Base) | Static Quality Filtered | 50K | 47.6% |
| 4 | Qwen3-14B (Base) | All Unclean Sample | 74K | 45.7% |
| 5 | DeepSeek-Coder-6.7B | Baseline | N/A | 45.7% |
| 6 | DeepSeek-Coder-6.7B | LLM-Quality Filtered | Original run sample | 37.8% |
| 7 | Qwen2.5-Coder-7B | All Unclean Sample | 74K | 36.0% |
| 8 | Qwen2.5-Coder-7B | Static Quality Filtered | 50K | 33.5% |
| 9 | Qwen3-14B (Base) | Baseline | N/A | 32.3% |
| 10 | Qwen2.5-Coder-3B | LLM-Quality Filtered | Original run sample | 31.1% |
| 11 | DeepSeek-Coder-6.7B | All Unclean Sample | 74K | 29.9% |
| 12 | DeepSeek-Coder-6.7B | Static Quality Filtered | 50K | 26.8% |
| 13 | Qwen2.5-Coder-3B | All Unclean Sample | 74K | 24.4% |
| 14 | Qwen2.5-Coder-3B | Static Quality Filtered | 50K | 23.8% |
| 15 | Qwen2.5-Coder-7B | Baseline | N/A | 20.7% |
| 16 | Qwen2.5-Coder-3B | Baseline | N/A | 19.5% |
| 17 | LLaMA-3.2-3B | LLM-Quality Filtered | Original run sample | 8.5% |
| 18 | LLaMA-3.2-3B | Static Quality Filtered | 50K | 5.5% |
| 19 | LLaMA-3.2-3B | All Unclean Sample | 74K | 3.7% |
| 20 | LLaMA-3.2-3B | Baseline | N/A | 1.2% |
