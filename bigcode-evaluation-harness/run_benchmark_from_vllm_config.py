#!/usr/bin/env python3
"""Run HumanEvalFix from an EditBench vLLM YAML config and notify on completion."""

import argparse
import errno
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_NOTIFY_SCRIPT = Path("/shared_workspace_mfs/aadi/Projects/notify_telegram.py")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to the EditBench vLLM YAML config.")
    parser.add_argument(
        "--task",
        default="humanevalfixtests-python",
        help="Benchmark task name to pass to run_vllm_humanevalfix.py.",
    )
    parser.add_argument(
        "--prompt",
        default="instruct",
        help="Prompt template name to pass to run_vllm_humanevalfix.py.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Hostname for the vLLM server.")
    parser.add_argument("--limit", type=int, default=None, help="Optional dataset limit.")
    parser.add_argument("--limit-start", type=int, default=0, help="Optional dataset start offset.")
    parser.add_argument("--n-samples", type=int, default=1, help="Samples per task.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature. Defaults to config temperature if present, otherwise 0.2.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p value. Defaults to config top_p if present, otherwise 0.95.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max completion tokens. Defaults to config max_tokens if present, otherwise 1024.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Request concurrency. Defaults to config max_workers if present, otherwise 8.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=300,
        help="Per-request timeout in seconds for the underlying benchmark runner.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries per request for the underlying benchmark runner.",
    )
    parser.add_argument(
        "--wait-timeout",
        type=int,
        default=600,
        help="Seconds to wait for the vLLM server to respond before failing.",
    )
    parser.add_argument(
        "--wait-interval",
        type=int,
        default=5,
        help="Seconds between readiness checks.",
    )
    parser.add_argument(
        "--output-dir",
        default="vllm_humanevalfix_runs",
        help="Output directory for generations and metrics.",
    )
    parser.add_argument(
        "--notify-script",
        default=str(DEFAULT_NOTIFY_SCRIPT),
        help="Path to the Telegram notify script.",
    )
    parser.add_argument(
        "--notify-python",
        default=sys.executable,
        help="Python executable to use for the Telegram notify script.",
    )
    parser.add_argument(
        "--no-notify",
        action="store_true",
        help="Disable Telegram notifications.",
    )
    parser.add_argument(
        "--generation-only",
        action="store_true",
        help="Generate only and skip benchmark execution.",
    )
    return parser.parse_args()


def load_config(config_path):
    with config_path.open() as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}, got {type(data).__name__}")
    return data


def get_required(config, key, config_path):
    value = config.get(key)
    if value in (None, ""):
        raise ValueError(f"Missing required key {key!r} in {config_path}")
    return value


def wait_for_server(base_url, timeout, interval):
    deadline = time.time() + timeout
    models_url = f"{base_url}/v1/models"
    last_error = None
    while time.time() < deadline:
        try:
            response = requests.get(models_url, timeout=min(interval, 10))
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            time.sleep(interval)
    raise RuntimeError(f"vLLM server at {models_url} was not ready within {timeout}s: {last_error}")


def build_run_stem(task, model, prompt, limit_start, end):
    return f"{task}__{model}__{prompt}__start{limit_start}__end{end}"


def find_output_end(output_dir, task, model, prompt, limit_start, explicit_limit):
    if explicit_limit is not None:
        return limit_start + explicit_limit

    prefix = f"{task}__{model}__{prompt}__start{limit_start}__end"
    candidates = sorted(
        list(output_dir.glob(f"{prefix}*.generations.json")) + list(output_dir.glob(f"{prefix}*.metrics.json")),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No benchmark outputs found in {output_dir} for prefix {prefix}")

    stem = candidates[0].stem
    end_str = stem.rsplit("__end", 1)[1].split(".", 1)[0]
    return int(end_str)


def rename_outputs(output_dir, source_stem, target_stem):
    renamed = []
    for suffix in (".generations.json", ".metrics.json", ".errors.json"):
        source = output_dir / f"{source_stem}{suffix}"
        target = output_dir / f"{target_stem}{suffix}"
        if source.exists():
            if source != target:
                if target.exists():
                    target.unlink()
                shutil.move(str(source), str(target))
            renamed.append(target)
    return renamed


def notify(args, message):
    if args.no_notify:
        return
    notify_script = Path(args.notify_script)
    if not notify_script.exists():
        print(f"[warn] notify script not found: {notify_script}", file=sys.stderr)
        return
    try:
        subprocess.run(
            [args.notify_python, str(notify_script), message],
            check=False,
            cwd=str(REPO_ROOT),
        )
    except Exception as exc:
        print(f"[warn] failed to invoke notify script: {exc}", file=sys.stderr)


def benchmark_message(model_name: str, state: str, score: str | None = None) -> str:
    parts = ["HumanEvalFix", model_name, state]
    if score:
        parts.append(score)
    return " | ".join(parts)


def summarize_metrics(metrics_path):
    if not metrics_path.exists():
        return "metrics unavailable"
    with metrics_path.open() as handle:
        metrics = json.load(handle)
    if not isinstance(metrics, dict):
        return json.dumps(metrics)
    ordered_keys = sorted(metrics)
    return ", ".join(f"{key}={metrics[key]}" for key in ordered_keys)


def write_json(path: Path, payload: dict):
    """Atomically write ``payload`` to ``path`` via a unique temp file.

    The master's refresh loop reads summary.json without locking; a non-atomic
    write_text races with those reads and can surface truncated JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        except OSError as exc:
            if exc.errno not in (errno.ENOLCK, errno.ENOSYS):
                raise
        tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        os.replace(tmp_path, path)


def update_summary(summary_path: Path, **updates) -> None:
    """Merge new fields into an existing summary.json and rewrite atomically."""
    data: dict = {}
    if summary_path.exists():
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8") or "{}")
        except json.JSONDecodeError:
            data = {}
    data.update(updates)
    write_json(summary_path, data)


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)

    port = get_required(config, "port", config_path)
    benchmark_model_name = get_required(config, "model_name", config_path)
    api_model_name = config.get("served_model_name") or benchmark_model_name
    base_url = f"http://{args.host}:{port}"

    temperature = args.temperature if args.temperature is not None else config.get("temperature", 0.2)
    top_p = args.top_p if args.top_p is not None else config.get("top_p", 0.95)
    max_tokens = args.max_tokens if args.max_tokens is not None else config.get("max_tokens", 1024)
    concurrency = args.concurrency if args.concurrency is not None else config.get("max_workers", 8)

    try:
        wait_for_server(base_url, args.wait_timeout, args.wait_interval)
    except Exception as exc:
        notify(args, benchmark_message(benchmark_model_name, "failed"))
        raise

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    detached_root = output_dir / "_detached"
    # Include a nanosecond suffix so concurrent servers sharing the same output
    # root cannot collide on the detached run directory.
    run_token = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{time.time_ns() % 1_000_000_000:09d}"
    run_name = f"{args.task}__{benchmark_model_name}__{args.prompt}__{run_token}"
    run_dir = detached_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    eval_log_path = run_dir / "detached_eval.log"

    # Seed summary.json BEFORE generation so the master scheduler can tell a
    # running job from a crashed one. The rest of this function updates the
    # same file through its lifecycle.
    base_summary = {
        "benchmark": "HumanEvalFix",
        "run_name": run_name,
        "status": "running_generation",
        "task": args.task,
        "prompt": args.prompt,
        "model_name": benchmark_model_name,
        "api_model": api_model_name,
        "config_path": str(config_path),
        "eval_log_path": str(eval_log_path),
        "generation_started_at_utc": datetime.now(timezone.utc).isoformat(),
        "limit_start": args.limit_start,
    }
    write_json(summary_path, base_summary)

    bench_script = REPO_ROOT / "run_vllm_humanevalfix.py"

    command = [
        sys.executable,
        str(bench_script),
        "--base-url",
        base_url,
        "--model",
        str(api_model_name),
        "--task",
        args.task,
        "--prompt",
        args.prompt,
        "--temperature",
        str(temperature),
        "--top-p",
        str(top_p),
        "--max-tokens",
        str(max_tokens),
        "--n-samples",
        str(args.n_samples),
        "--concurrency",
        str(concurrency),
        "--request-timeout",
        str(args.request_timeout),
        "--retries",
        str(args.retries),
        "--output-dir",
        str(run_dir),
        "--limit-start",
        str(args.limit_start),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    command.append("--generation-only")

    env = os.environ.copy()
    env.setdefault("HF_ALLOW_CODE_EVAL", "1")

    try:
        subprocess.run(command, cwd=str(REPO_ROOT), env=env, check=True)
    except subprocess.CalledProcessError as exc:
        update_summary(
            summary_path,
            status="generation_failed",
            error=f"generation exit code {exc.returncode}",
            generation_failed_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        notify(args, benchmark_message(benchmark_model_name, "failed"))
        raise

    try:
        end = find_output_end(
            output_dir=run_dir,
            task=args.task,
            model=api_model_name,
            prompt=args.prompt,
            limit_start=args.limit_start,
            explicit_limit=args.limit,
        )
    except Exception as exc:
        update_summary(
            summary_path,
            status="generation_failed",
            error=f"outputs missing after generation: {exc}",
            generation_failed_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        notify(args, benchmark_message(benchmark_model_name, "failed"))
        raise

    source_stem = build_run_stem(args.task, api_model_name, args.prompt, args.limit_start, end)
    target_stem = build_run_stem(args.task, benchmark_model_name, args.prompt, args.limit_start, end)
    try:
        rename_outputs(run_dir, source_stem, target_stem)
    except Exception as exc:
        update_summary(
            summary_path,
            status="generation_failed",
            error=f"rename_outputs failed: {exc}",
            generation_failed_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        notify(args, benchmark_message(benchmark_model_name, "failed"))
        raise

    # Verify the renamed generation file actually exists. If for any reason
    # the rename left us in a half-state (e.g. target absent and source also
    # absent) fail cleanly and mark the summary rather than writing a
    # summary that points at a nonexistent file.
    generations_path = run_dir / f"{target_stem}.generations.json"
    metrics_path = run_dir / f"{target_stem}.metrics.json"
    errors_path = run_dir / f"{target_stem}.errors.json"
    fallback_generations = run_dir / f"{source_stem}.generations.json"
    if not generations_path.exists() and fallback_generations.exists():
        # Rename didn't happen (e.g. source == target); fall back.
        generations_path = fallback_generations
        metrics_path = run_dir / f"{source_stem}.metrics.json"
        errors_path = run_dir / f"{source_stem}.errors.json"
    if not generations_path.exists():
        err = f"missing generations file after rename: {generations_path}"
        update_summary(
            summary_path,
            status="generation_failed",
            error=err,
            generation_failed_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        notify(args, benchmark_message(benchmark_model_name, "failed"))
        raise FileNotFoundError(err)

    update_summary(
        summary_path,
        status="generation_complete",
        generation_path=str(generations_path),
        metrics_path=str(metrics_path),
        error_path=str(errors_path),
        generation_completed_at_utc=datetime.now(timezone.utc).isoformat(),
        limit_end=end,
    )

    if args.generation_only:
        notify(args, benchmark_message(benchmark_model_name, "generation_only"))
        print(f"Generation finished for config: {config_path}")
        print(f"Outputs: {generations_path}")
        return

    detached_worker = REPO_ROOT / "detached_eval_worker.py"
    with eval_log_path.open("w", encoding="utf-8") as log_handle:
        proc = subprocess.Popen(
            [
                sys.executable,
                str(detached_worker),
                "--run-dir",
                str(run_dir),
                "--output-root",
                str(output_dir),
                "--summary-path",
                str(summary_path),
                "--notify-script",
                args.notify_script,
            ],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )

    # Record the detached worker pid so master / ops tooling can check
    # liveness later instead of relying only on wall-clock staleness.
    update_summary(summary_path, detached_worker_pid=proc.pid)

    notify(args, benchmark_message(benchmark_model_name, "eval_pending"))

    print(f"Generation finished for config: {config_path}")
    print(f"Model: {benchmark_model_name} (API model: {api_model_name})")
    print(f"Base URL: {base_url}")
    print(f"Outputs: {generations_path}")
    print(f"Detached eval log: {eval_log_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
