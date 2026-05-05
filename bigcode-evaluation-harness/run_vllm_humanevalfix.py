#!/usr/bin/env python3
"""Run HumanEvalFix via a vLLM OpenAI-compatible server."""

import argparse
import importlib.util
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_humanevalpack_module():
    module_path = REPO_ROOT / "bigcode_eval" / "tasks" / "humanevalpack.py"
    spec = importlib.util.spec_from_file_location("humanevalpack_local", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


HUMANEVALPACK = load_humanevalpack_module()
FIX_TASKS = {
    name: task_class
    for name, task_class in HUMANEVALPACK.create_all_tasks().items()
    if name.startswith("humanevalfix")
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8788",
        help="Base URL for the vLLM OpenAI-compatible server.",
    )
    parser.add_argument(
        "--model",
        default="clean-edit-llama-3.2-3b",
        help="Model ID exposed by the vLLM API.",
    )
    parser.add_argument(
        "--task",
        default="humanevalfixtests-python",
        choices=sorted(FIX_TASKS),
        help="HumanEvalFix task variant to run.",
    )
    parser.add_argument(
        "--prompt",
        default="instruct",
        help="Prompt template name from bigcode_eval.tasks.humanevalpack.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of benchmark items.")
    parser.add_argument("--limit-start", type=int, default=0, help="Start offset into the benchmark split.")
    parser.add_argument("--n-samples", type=int, default=1, help="Number of samples per problem.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling parameter.")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum completion tokens.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Maximum in-flight requests to the vLLM server.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=300,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries per request on transient failures.",
    )
    parser.add_argument(
        "--output-dir",
        default="vllm_humanevalfix_runs",
        help="Directory for generations and metrics.",
    )
    parser.add_argument(
        "--generation-only",
        action="store_true",
        help="Generate only and skip benchmark execution.",
    )
    return parser.parse_args()


def normalize_base_url(base_url):
    return base_url.rstrip("/")


def get_task(task_name, prompt):
    task_class = FIX_TASKS[task_name]
    return task_class(prompt=prompt)


def fetch_available_models(base_url, timeout):
    response = requests.get(f"{base_url}/v1/models", timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    return [item["id"] for item in payload.get("data", [])]


def generate_one(base_url, model, prompt, stop, n_samples, temperature, top_p, max_tokens, timeout, retries):
    url = f"{base_url}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n": n_samples,
        "stop": stop,
    }
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            choices = sorted(data["choices"], key=lambda item: item["index"])
            return [choice.get("text", "") for choice in choices]
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2 * attempt, 5))
    raise RuntimeError(f"Request failed after {retries} attempts: {last_error}") from last_error


def main():
    args = parse_args()
    base_url = normalize_base_url(args.base_url)
    task = get_task(args.task, args.prompt)
    dataset = task.get_dataset()

    start = args.limit_start
    end = len(dataset) if args.limit is None else min(len(dataset), start + args.limit)
    if start >= len(dataset):
        raise ValueError(f"--limit-start={start} is outside the dataset of size {len(dataset)}")

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_stem = f"{args.task}__{args.model}__{args.prompt}__start{start}__end{end}"
    generations_path = output_dir / f"{run_stem}.generations.json"
    metrics_path = output_dir / f"{run_stem}.metrics.json"
    errors_path = output_dir / f"{run_stem}.errors.json"

    available_models = fetch_available_models(base_url, args.request_timeout)
    if args.model not in available_models:
        raise ValueError(
            f"Model {args.model!r} not exposed by {base_url}/v1/models. "
            f"Available models: {available_models}"
        )

    docs = [dataset[idx] for idx in range(start, end)]
    references = [task.get_reference(doc) for doc in docs]
    generations = [None] * len(docs)
    request_errors = []

    print(
        f"Running {args.task} on {len(docs)} problems via {base_url} "
        f"with model={args.model}, prompt={args.prompt}, n_samples={args.n_samples}"
    )

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {}
        for position, doc_idx in enumerate(range(start, end)):
            prompt = task.get_prompt(dataset[doc_idx])
            futures[
                executor.submit(
                    generate_one,
                    base_url,
                    args.model,
                    prompt,
                    task.stop_words,
                    args.n_samples,
                    args.temperature,
                    args.top_p,
                    args.max_tokens,
                    args.request_timeout,
                    args.retries,
                )
            ] = (position, doc_idx)

        for future in as_completed(futures):
            position, doc_idx = futures[future]
            task_id = dataset[doc_idx]["task_id"]
            try:
                completions = future.result()
                generations[position] = [
                    task.postprocess_generation(task.get_prompt(dataset[doc_idx]) + text, doc_idx)
                    for text in completions
                ]
                print(f"[ok] {task_id}")
            except Exception as exc:
                generations[position] = [""] * args.n_samples
                request_errors.append({"task_id": task_id, "index": doc_idx, "error": str(exc)})
                print(f"[error] {task_id}: {exc}")

    with generations_path.open("w") as handle:
        json.dump(generations, handle, indent=2)
    if request_errors:
        with errors_path.open("w") as handle:
            json.dump(request_errors, handle, indent=2)

    print(f"Saved generations to {generations_path}")
    if request_errors:
        print(f"Saved request errors to {errors_path}")

    if args.generation_only:
        return

    results = task.process_results(generations, references)
    with metrics_path.open("w") as handle:
        json.dump(results, handle, indent=2)
    print(f"Saved metrics to {metrics_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
