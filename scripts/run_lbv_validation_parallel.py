"""Run checkpointed LBV validation across deterministic subprocess shards."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--mechanism", type=Path, required=True)
    parser.add_argument("--mechanism-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--case-timeout-s", type=float, default=300.0)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    design = pd.read_csv(args.design)
    shard_dir = Path("data/processed/lbv_worker_shards") / args.mechanism_id
    log_dir = Path("results/logs/lbv_workers") / args.mechanism_id
    result_dir = Path("results/raw/lbv_worker_shards") / args.mechanism_id
    for directory in (shard_dir, log_dir, result_dir, args.output.parent):
        directory.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str((Path.cwd() / "src").resolve())
    processes: list[tuple[subprocess.Popen[str], object, object, Path]] = []
    shard_outputs: list[Path] = []
    for index in range(args.workers):
        shard = design.iloc[index::args.workers].copy()
        shard_design = shard_dir / f"shard_{index}.csv"
        shard_output = result_dir / f"shard_{index}.csv"
        stdout_path = log_dir / f"shard_{index}.stdout.log"
        stderr_path = log_dir / f"shard_{index}.stderr.log"
        shard.to_csv(shard_design, index=False)
        stdout = stdout_path.open("w", encoding="utf-8")
        stderr = stderr_path.open("w", encoding="utf-8")
        command = [
            sys.executable,
            "scripts/run_lbv_validation_with_timeout.py",
            "--design", str(shard_design),
            "--mechanism", str(args.mechanism),
            "--mechanism-id", args.mechanism_id,
            "--output", str(shard_output),
            "--case-timeout-s", str(args.case_timeout_s),
            "--resume",
        ]
        if args.retry_failed:
            command.append("--retry-failed")
        process = subprocess.Popen(
            command, cwd=Path.cwd(), env=env, stdout=stdout, stderr=stderr, text=True
        )
        processes.append((process, stdout, stderr, shard_output))
        shard_outputs.append(shard_output)

    failures: list[str] = []
    for index, (process, stdout, stderr, shard_output) in enumerate(processes):
        returncode = process.wait()
        stdout.close()
        stderr.close()
        if returncode != 0:
            failures.append(f"worker {index} exited {returncode}: {shard_output}")
    if failures:
        raise SystemExit("; ".join(failures))

    result = pd.concat([pd.read_csv(path) for path in shard_outputs], ignore_index=True)
    if result.design_id.duplicated().any():
        raise SystemExit("duplicate design IDs in worker outputs")
    expected = set(design.design_id)
    observed = set(result.design_id)
    if expected != observed:
        raise SystemExit(
            f"incomplete worker outputs: missing={len(expected - observed)} "
            f"unknown={len(observed - expected)}"
        )
    order = {design_id: index for index, design_id in enumerate(design.design_id)}
    result["_order"] = result.design_id.map(order)
    result = result.sort_values("_order").drop(columns="_order")
    result.to_csv(args.output, index=False)
    print(result.groupby("status").size().to_string())
    print(args.output)


if __name__ == "__main__":
    main()
