"""Convert the ReSpecTh NH3 RKD archive to analysis-ready CSV tables."""

from __future__ import annotations

import argparse
from pathlib import Path

from pca_ensemble.rkd import parse_rkd_directory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    data, audit = parse_rkd_directory(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / "respecth_nh3_long.csv"
    audit_path = args.output_dir / "respecth_nh3_parse_audit.csv"
    coverage_path = args.output_dir / "respecth_nh3_coverage.csv"
    data.to_csv(data_path, index=False)
    audit.to_csv(audit_path, index=False)
    coverage = (
        data.groupby(["experiment_type", "observable"], dropna=False)
        .agg(rows=("dataset_id", "size"), campaigns=("campaign_id", "nunique"),
             sources=("source_file", "nunique"), min_T_K=("temperature_K", "min"),
             max_T_K=("temperature_K", "max"), min_P_Pa=("pressure_Pa", "min"),
             max_P_Pa=("pressure_Pa", "max"), min_alpha=("cracking_ratio", "min"),
             max_alpha=("cracking_ratio", "max"))
        .reset_index()
    )
    coverage.to_csv(coverage_path, index=False)
    print(f"parsed_files={int((audit.status == 'parsed').sum())}")
    print(f"failed_files={int((audit.status == 'failed').sum())}")
    print(f"rows={len(data)} campaigns={data.campaign_id.nunique()}")
    print(data_path)
    print(audit_path)
    print(coverage_path)


if __name__ == "__main__":
    main()

