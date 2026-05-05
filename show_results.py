#!/usr/bin/env python3
"""Display HumanEvalFix benchmark results in a tabular format.

Reads all *.metrics.json files from the results directory and presents
pass@1 scores grouped by model with columns for each ablation.
"""

import csv
import json
import glob
import os
import re
from collections import defaultdict

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "bigcode-evaluation-harness", "vllm_humanevalfix_runs",
)

# Only consider full runs (end164), skip partial/test runs and _detached duplicates
PATTERN = os.path.join(RESULTS_DIR, "humanevalfixtests-python__*__instruct__start0__end164.metrics.json")

# Ablation keywords to match in model name suffixes
ABLATIONS = ["baseline", "clean", "dirty", "unclean74k"]
ABLATION_LABELS = ["Baseline", "LLM-Cleaning", "Static-Cleaning", "Unfiltered"]

MODEL_TYPE_MAP = {
    "deepseek-coder-6.7b": "instruct",
    "deepseek-coder-6.7b-base": "base",
    "llama-3.2-3b": "base",
    "qwen2.5-3b": "base",
    "qwen2.5-coder-3b": "base",
    "qwen2.5-coder-7b": "base",
    "qwen2.5-coder-14b": "base",
    "qwen2.5-coder-3b-instruct": "instruct",
    "qwen2.5-coder-7b-instruct": "instruct",
    "qwen2.5-coder-14b-instruct": "instruct",
    "qwen3-4b-base": "base",
    "qwen3-8b-base": "base",
    "qwen3-14b-base": "base",
    "starcoder2-3b": "base",
    "starcoder2-7b": "base",
    "starcoder2-15b": "base",
}

# Models where the standalone file uses "-base" as a baseline indicator,
# NOT as part of the model name. Their LoRA variants omit "-base".
# e.g. "llama-3.2-3b-base" is the baseline of "llama-3.2-3b"
#      but "deepseek-coder-6.7b-base" IS the model name (has its own -lora- variants)
BASELINE_ALIAS_MAP = {
    "llama-3.2-3b-base": "llama-3.2-3b",
    "qwen2.5-coder-3b-base": "qwen2.5-coder-3b",
    "qwen2.5-coder-7b-base": "qwen2.5-coder-7b",
}


def classify(model_name: str):
    """Return (base_model, ablation) from a model name string."""
    name = model_name

    # Handle edit-model pattern: {clean,dirty}-edit-<model>
    m = re.match(r"^(clean|dirty)-edit-(.+)$", name)
    if m:
        return m.group(2), m.group(1)

    # Handle lora variants
    m2 = re.match(r"^(.+)-lora-(unclean74k|clean|dirty)$", name)
    if m2:
        return m2.group(1), m2.group(2)

    # Handle explicit -baseline suffix
    if name.endswith("-baseline"):
        return name[: -len("-baseline")], "baseline"

    # Handle models where "-base" suffix is a baseline indicator, not part of the name
    if name in BASELINE_ALIAS_MAP:
        return BASELINE_ALIAS_MAP[name], "baseline"

    # Fallback: treat as baseline
    return name, "baseline"


def get_model_type(base_model: str) -> str:
    """Return 'instruct' or 'base' from the hashmap."""
    if base_model in MODEL_TYPE_MAP:
        return MODEL_TYPE_MAP[base_model]
    if "instruct" in base_model:
        return "instruct"
    if "-base" in base_model:
        return "base"
    return "base"


TOTAL_PROBLEMS = 164


def _scan_files(extension: str):
    """Scan all top-level result files with the given extension.

    Returns {(base_model, ablation): filepath}.
    Prefers the largest range; on ties the last in sorted order wins.
    """
    found = {}  # key -> (path, range_size)
    paths = sorted(glob.glob(os.path.join(
        RESULTS_DIR,
        f"humanevalfixtests-python__*__instruct__start*__end*.{extension}",
    )))
    for path in paths:
        fname = os.path.basename(path)
        parts = fname.split("__")
        model_name = parts[1]
        range_part = parts[3]  # e.g. "start0"
        end_part = parts[4].split(".")[0]  # e.g. "end164"
        start = int(range_part.replace("start", ""))
        end = int(end_part.replace("end", ""))
        base, ablation = classify(model_name)
        key = (base, ablation)
        size = end - start
        if key not in found or size >= found[key][1]:
            found[key] = (path, size)
    return {k: v[0] for k, v in found.items()}


def _count_gen_entries(gen_path: str) -> tuple:
    """Return (total_entries, non_empty_entries) from a generations file."""
    with open(gen_path) as f:
        data = json.load(f)
    total = len(data)
    non_empty = sum(
        1 for entry in data
        if entry and isinstance(entry, list) and entry[0] and str(entry[0]).strip()
    )
    return total, non_empty


def _load_single(gen_files, met_files, ablation_set):
    """Load results for a specific set of ablations.

    Returns (results, gen_status, eval_status) dicts filtered to ablation_set.
    """
    results = defaultdict(dict)
    gen_status = defaultdict(dict)
    eval_status = defaultdict(dict)

    all_keys = set(gen_files.keys()) | set(met_files.keys())

    for base, ablation in all_keys:
        if ablation not in ablation_set:
            continue

        if (base, ablation) in gen_files:
            total, non_empty = _count_gen_entries(gen_files[(base, ablation)])
            pct = non_empty / TOTAL_PROBLEMS * 100
            gen_status[base][ablation] = f"{pct:.0f}%"
        else:
            gen_status[base][ablation] = "No"

        if (base, ablation) in met_files:
            with open(met_files[(base, ablation)]) as f:
                data = json.load(f)
            score = data.get("pass@1")
            if score is not None:
                results[base][ablation] = score
                eval_status[base][ablation] = "Yes"
            else:
                eval_status[base][ablation] = "No"
        else:
            eval_status[base][ablation] = "No"

    return results, gen_status, eval_status


def load_results():
    """Load main experiment results.

    Returns (results, gen_status, eval_status) dicts.
    """
    gen_files = _scan_files("generations.json")
    met_files = _scan_files("metrics.json")

    return _load_single(gen_files, met_files, set(ABLATIONS))


def _missing_summary(results, status_dict, model, label_fn):
    """Build a summary string for missing ablations."""
    parts = []
    for abl, abl_label in zip(ABLATIONS, ABLATION_LABELS):
        if abl in results.get(model, {}):
            continue  # has a score, not missing
        val = status_dict.get(model, {}).get(abl)
        mapped = label_fn(val)
        if mapped:
            parts.append(f"{abl_label}({mapped})")
    return "; ".join(parts) if parts else ""


def print_table(results, gen_status, eval_status, ablations, ablation_labels, title=None, model_filter=None):
    """Print a formatted table of results."""
    all_models = sorted(
        m for m in (set(results) | set(gen_status) | set(eval_status))
        if model_filter is None or m in model_filter
    )
    if not all_models:
        return

    model_w = max((len(m) for m in all_models), default=20)
    model_w = max(model_w, len("Model"))
    type_w = 10
    col_w = max(12, max((len(l) for l in ablation_labels), default=12) + 2)
    extra_w = 35

    if title:
        print(f"\n{'=' * 40}")
        print(f"  {title}")
        print(f"{'=' * 40}")

    header = (
        f"{'Model':<{model_w}}  {'Type':<{type_w}}  "
        + "  ".join(f"{a:>{col_w}}" for a in ablation_labels)
        + f"  {'Not Generated':<{extra_w}}  {'Gen. Not Evald':<{extra_w}}"
    )
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)

    for model in all_models:
        mtype = get_model_type(model)
        row = f"{model:<{model_w}}  {mtype:<{type_w}}  "
        cells = []
        for abl in ablations:
            score = results.get(model, {}).get(abl)
            if score is not None:
                cells.append(f"{score * 100:>{col_w}.1f}%")
            else:
                cells.append(f"{'-':>{col_w}}")
        row += "  ".join(cells)

        not_gen = []
        gen_not_eval = []
        for abl, abl_label in zip(ablations, ablation_labels):
            if abl in results.get(model, {}):
                continue
            gs = gen_status.get(model, {}).get(abl, "No")
            es = eval_status.get(model, {}).get(abl, "No")
            if gs == "No":
                not_gen.append(abl_label)
            else:
                if es != "Yes":
                    gen_not_eval.append(f"{abl_label}({gs})" if gs != "100%" else abl_label)

        not_gen_str = "; ".join(not_gen) if not_gen else ""
        gen_not_eval_str = "; ".join(gen_not_eval) if gen_not_eval else ""
        row += f"  {not_gen_str:<{extra_w}}  {gen_not_eval_str:<{extra_w}}"
        print(row)

    print(sep)
    print(f"\nTotal models: {len(all_models)}")
    total_runs = sum(len(v) for v in results.values())
    print(f"Total runs:   {total_runs}")


def save_csv(results, gen_status, eval_status, ablations, ablation_labels, path, model_filter=None):
    """Save results to CSV."""
    all_models = sorted(
        m for m in (set(results) | set(gen_status) | set(eval_status))
        if model_filter is None or m in model_filter
    )
    if not all_models:
        return
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Model", "Type"] + ablation_labels + ["Not Generated", "Gen. Not Eval'd"]
        )
        for model in all_models:
            mtype = get_model_type(model)
            row = [model, mtype]
            for abl in ablations:
                score = results.get(model, {}).get(abl)
                row.append(f"{score * 100:.1f}" if score is not None else "-")

            not_gen = []
            gen_not_eval = []
            for abl, abl_label in zip(ablations, ablation_labels):
                if abl in results.get(model, {}):
                    continue
                gs = gen_status.get(model, {}).get(abl, "No")
                es = eval_status.get(model, {}).get(abl, "No")
                if gs == "No":
                    not_gen.append(abl_label)
                else:
                    if es != "Yes":
                        gen_not_eval.append(
                            f"{abl_label}({gs})" if gs != "100%" else abl_label
                        )

            row.append("; ".join(not_gen) if not_gen else "")
            row.append("; ".join(gen_not_eval) if gen_not_eval else "")
            writer.writerow(row)
    print(f"\nCSV saved to: {path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    main_res, main_gen, main_eval = load_results()

    print_table(main_res, main_gen, main_eval, ABLATIONS, ABLATION_LABELS,
                title="MAIN RESULTS")

    save_csv(main_res, main_gen, main_eval, ABLATIONS, ABLATION_LABELS,
             os.path.join(base_dir, "results.csv"))
