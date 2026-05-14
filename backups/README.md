# HumanEvalFix Benchmark-Run Backups

This directory holds per-run `tar.gz` archives of canonical HEF runs.
Each archive expands into a single run dir under
`bigcode-evaluation-harness/vllm_humanevalfix_runs/_detached/<run-name>/`.

## What's inside each `<run>.tar.gz`

```
<run-name>/
├── summary.json                          ← state + pass_at_1
├── metrics.json                          ← pass@1 (either flat {pass@1:...}
│                                            or task-keyed {<task>: {pass@1:...}})
├── <run-name>.generations.json[l[.gz]]   ← gzipped JSONL on new runs,
│                                            plain JSON on legacy ones
└── partials.jsonl                        ← per-task resume checkpoint (small)
```

Eval logs, the `harness_input.json` converter artifact, and the
`summary.json.lock` advisory file are **not** included.

## Quick-start: restore on a fresh machine

```bash
git clone git@github.com:Aaditya-Bhatia/human-eval-fix.git
cd human_eval_fix

# Restore one run
mkdir -p bigcode-evaluation-harness/vllm_humanevalfix_runs/_detached
tar -xzf backups/<run-name>.tar.gz \
    -C bigcode-evaluation-harness/vllm_humanevalfix_runs/_detached/

# Restore everything (paired with the Master_VLLM extract script)
python3 /path/to/Master_VLLM/scripts/backup_extract.py --repo-root .
```

If you're on a machine with different paths than the original CPU
host, also rewrite the paths embedded in `summary.json`:

```bash
python3 /path/to/Master_VLLM/scripts/backup_extract.py --repo-root . \
  --rewrite-paths-from /shared_workspace_mfs/aadi/Projects \
  --rewrite-paths-to   /your/local/Projects
```

## Adding a new run to the backup

```bash
cd /path/to/Master_VLLM
python3 scripts/backup_compress.py --include-quarantine --execute --only-benchmark HumanEvalFix
cd /path/to/human_eval_fix
git add backups/<new-run>.tar.gz backups/INDEX.csv
git commit -m "Back up <model>-<variant> HEF run"
git push origin main                # NOT 'bigcode' — see note below
```

## Dual-remote note

This repo has two remotes with **unrelated histories**:
- `origin` → `human-eval-fix.git` (this CPU host's tree; backups go here)
- `bigcode` → `bigcode-evaluation-harness.git` (GPU pod's fork; pull only)

Always push backups to `origin`. Pushing to `bigcode` will fail with
`refusing to merge unrelated histories`. See
`Master_VLLM/docs/repo_layout.md` for the full story.

## INDEX.csv

`INDEX.csv` carries one row per tarball with run name, model, variant,
file count, raw on-disk size, and compressed tarball size.

## See also

- Full procedure: `Master_VLLM/docs/BACKUP_LAYOUT.md`
- Cross-repo layout + path-drift handling: `Master_VLLM/docs/repo_layout.md`
- Run-classification rules (same-size vs different-size): `Master_VLLM/.claude/agents/results-pipeline-guide.md`
