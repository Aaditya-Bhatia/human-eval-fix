#!/usr/bin/env python3
"""Detached HumanEvalFix evaluator for pre-generated outputs."""

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

from run_vllm_humanevalfix import get_task


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--notify-script", default="")
    return parser.parse_args()


def update_summary(summary_path: Path, **updates):
    """Merge-update ``summary_path`` atomically under an advisory file lock.

    Readers (e.g. Master_VLLM's refresh loop) read the summary without
    locking, so the writer uses a unique temp file + rename to guarantee no
    truncated state is ever visible.
    """
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = summary_path.with_name(summary_path.name + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        except OSError as exc:
            if exc.errno not in (errno.ENOLCK, errno.ENOSYS):
                raise
        data = {}
        if summary_path.exists():
            try:
                raw = summary_path.read_text(encoding="utf-8").strip()
                data = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                data = {}
        data.update(updates)
        tmp_path = summary_path.with_name(
            f".{summary_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        os.replace(tmp_path, summary_path)


def load_summary(summary_path: Path) -> dict:
    return json.loads(summary_path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict | list):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def send_telegram(notify_script: str, message: str):
    if not notify_script:
        return
    path = Path(notify_script)
    if not path.exists():
        return
    subprocess.run([sys.executable, str(path), message], check=False)


def format_percent(value: object) -> str | None:
    if isinstance(value, (int, float)):
        return f"{float(value) * 100:.2f}%"
    return None


def benchmark_message(model_name: str, state: str, score: str | None = None) -> str:
    parts = ["HumanEvalFix", model_name, state]
    if score:
        parts.append(score)
    return " | ".join(parts)


def copy_if_exists(src: Path, dst: Path):
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    output_root = Path(args.output_root).resolve()
    summary_path = Path(args.summary_path).resolve()

    try:
        summary = load_summary(summary_path)
        generation_path = Path(summary["generation_path"]).resolve()
        metrics_path = Path(summary["metrics_path"]).resolve()
        error_path = Path(summary["error_path"]).resolve()
        task_name = str(summary["task"])
        prompt = str(summary["prompt"])
        model_name = str(summary["model_name"])
        limit_start = int(summary["limit_start"])
        limit_end = int(summary["limit_end"])

        update_summary(
            summary_path,
            status="eval_running",
            eval_started_at_utc=datetime.now(timezone.utc).isoformat(),
            detached_worker_pid=os.getpid(),
        )

        task = get_task(task_name, prompt)
        dataset = task.get_dataset()
        docs = [dataset[idx] for idx in range(limit_start, limit_end)]
        references = [task.get_reference(doc) for doc in docs]
        generations = json.loads(generation_path.read_text(encoding="utf-8"))
        metrics = task.process_results(generations, references)
        write_json(metrics_path, metrics)

        copy_if_exists(generation_path, output_root / generation_path.name)
        copy_if_exists(metrics_path, output_root / metrics_path.name)
        copy_if_exists(error_path, output_root / error_path.name)

        update_summary(
            summary_path,
            status="eval_complete",
            eval_completed_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        send_telegram(
            args.notify_script,
            benchmark_message(model_name, "done", format_percent(metrics.get("pass@1"))),
        )
    except Exception as exc:
        update_summary(
            summary_path,
            status="eval_failed",
            error=str(exc),
            eval_failed_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        send_telegram(args.notify_script, benchmark_message(model_name if 'model_name' in locals() else run_dir.name, "failed"))
        raise


if __name__ == "__main__":
    main()
