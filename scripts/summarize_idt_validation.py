"""Summarize exact-criterion IDT validation without claiming cross-validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/processed"))
    args = parser.parse_args()
    data = pd.concat([pd.read_csv(path) for path in args.inputs], ignore_index=True)
    completed = data[data.status.eq("completed")].copy()
    summaries = []
    for mechanism, frame in completed.groupby("mechanism_id"):
        log_residual = np.log10(frame.simulated_s / frame.experimental_s)
        summaries.append({
            "mechanism_id": mechanism, "n": len(frame),
            "campaigns": frame.campaign_id.nunique(),
            "mean_absolute_log10_error": float(np.abs(log_residual).mean()),
            "median_absolute_log10_error": float(np.abs(log_residual).median()),
            "root_mean_squared_log10_error": float(np.sqrt(np.mean(log_residual**2))),
            "mean_signed_log10_bias": float(log_residual.mean()),
            "geometric_bias_ratio": float(10 ** log_residual.mean()),
            "median_runtime_s": float(frame.runtime_s.median()),
            "external_cv_supported": frame.campaign_id.nunique() >= 2,
        })
    predictions = completed.pivot(index="dataset_id", columns="mechanism_id", values="simulated_s")
    metadata = (completed.drop_duplicates("dataset_id")
                .set_index("dataset_id")[["campaign_id", "doi", "experimental_s",
                                          "temperature_K", "pressure_Pa", "definition"]])
    common = metadata.join(predictions, how="inner").dropna()
    mechanism_columns = list(predictions.columns)
    common["equal_log_ensemble_s"] = np.exp(np.log(common[mechanism_columns]).mean(axis=1))
    common["equal_log_ensemble_abs_log10_error"] = np.abs(
        np.log10(common.equal_log_ensemble_s / common.experimental_s)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(args.output_dir / "idt_model_summary.csv", index=False)
    common.reset_index().to_csv(args.output_dir / "idt_common_predictions.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False))
    print("IDT stacking is not fitted because the exact-criterion subset contains one campaign.")


if __name__ == "__main__":
    main()

