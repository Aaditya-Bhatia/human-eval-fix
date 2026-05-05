## HumanEvalFix install

Installed on 2026-04-04 (UTC).

### Local folders

- `octopack/`
  - Source repository for the OctoPack paper and HumanEvalPack benchmark materials.
- `bigcode-evaluation-harness/`
  - Evaluation runner with `HumanEvalFix` task definitions.
- `humanevalpack/`
  - Hugging Face dataset snapshot for `bigcode/humanevalpack`.

### HumanEvalFix task names

- `humanevalfixtests-python`
- `humanevalfixdocs-python`

Other languages available in the dataset and harness:

- `cpp`
- `go`
- `java`
- `js`
- `rust`

### Useful local paths

- `bigcode-evaluation-harness/bigcode_eval/tasks/humanevalpack.py`
- `bigcode-evaluation-harness/docs/README.md`
- `octopack/README.md`
- `humanevalpack/python/test-00000-of-00001.parquet`

### Example run

```bash
cd /shared_workspace_mfs/aadi/Projects/bigcode-evaluation-harness
accelerate launch main.py \
  --model <MODEL_NAME> \
  --prompt <PROMPT> \
  --tasks humanevalfixtests-python \
  --temperature 0.2 \
  --n_samples 20 \
  --batch_size 10 \
  --allow_code_execution
```
