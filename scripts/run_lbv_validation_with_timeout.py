"""Run LBV validation with an auditable per-case subprocess timeout."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd


def timeout_row(record: pd.Series, mechanism_id: str, timeout_s: float) -> dict[str, object]:
    return {
        "design_id": record.design_id,
        "dataset_id": record.dataset_id,
        "campaign_id": record.campaign_id,
        "doi": record.doi,
        "mechanism_id": mechanism_id,
        "experimental_m_per_s": record.value,
        "temperature_K": record.temperature_K,
        "pressure_Pa": record.pressure_Pa,
        "equivalence_ratio": record.equivalence_ratio,
        "cracking_ratio": record.cracking_ratio,
        "apparatus": record.apparatus,
        "status": "failed",
        "simulated_m_per_s": np.nan,
        "relative_error": np.nan,
        "converged": False,
        "failure_reason": f"TimeoutError: exceeded {timeout_s:g} s",
        "grid_points": 0,
        "transport_model": "multicomponent",
        "soret_enabled": True,
        "runtime_s": timeout_s,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--mechanism", type=Path, required=True)
    parser.add_argument("--mechanism-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-timeout-s", type=float, default=600.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="When resuming, retain completed rows and recompute all other statuses.",
    )
    args = parser.parse_args()

    design = pd.read_csv(args.design)
    rows: list[dict[str, object]] = []
    done: set[str] = set()
    if args.resume and args.output.exists():
        previous = pd.read_csv(args.output)
        if args.retry_failed:
            previous = previous[previous.status.eq("completed")].copy()
        rows = previous.to_dict("records")
        done = set(previous.design_id)

    workspace = Path("results/logs/lbv_case_workspace") / args.mechanism_id
    workspace.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str((Path.cwd() / "src").resolve())

    for _, record in design.iterrows():
        if record.design_id in done:
            continue
        case_design = workspace / f"{record.design_id}_design.csv"
        case_output = workspace / f"{record.design_id}_result.csv"
        case_log = workspace / f"{record.design_id}.log"
        pd.DataFrame([record]).to_csv(case_design, index=False)
        command = [
            sys.executable,
            "scripts/run_lbv_validation.py",
            "--design", str(case_design),
            "--mechanism", str(args.mechanism),
            "--mechanism-id", args.mechanism_id,
            "--output", str(case_output),
            "--max-cases", "1",
        ]
        start = perf_counter()
        try:
            completed = subprocess.run(
                command, cwd=Path.cwd(), env=env, capture_output=True, text=True,
                timeout=args.case_timeout_s, check=False,
            )
            elapsed = perf_counter() - start
            case_log.write_text(
                f"returncode={completed.returncode}\nruntime_s={elapsed:.6f}\n"
                f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}",
                encoding="utf-8",
            )
            if completed.returncode == 0 and case_output.exists():
                result = pd.read_csv(case_output).iloc[0].to_dict()
                rows.append(result)
            else:
                failed = timeout_row(record, args.mechanism_id, elapsed)
                failed["failure_reason"] = f"ChildProcessError: return code {completed.returncode}"
                rows.append(failed)
        except subprocess.TimeoutExpired as exc:
            elapsed = perf_counter() - start
            stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            case_log.write_text(
                f"returncode=timeout\nruntime_s={elapsed:.6f}\n"
                f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
                encoding="utf-8",
            )
            rows.append(timeout_row(record, args.mechanism_id, args.case_timeout_s))
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(record.design_id, rows[-1]["status"], f"{elapsed:.2f}s", flush=True)

    result = pd.DataFrame(rows)
    print(result.groupby("status").size().to_string())
    print(args.output)


if __name__ == "__main__":
    main()
