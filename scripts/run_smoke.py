"""Run a traceable 0D Cantera smoke case; this is not publication evidence."""

from __future__ import annotations

import json
from pathlib import Path

from pca_ensemble.io import load_yaml
from pca_ensemble.mechanisms import audit_mechanism, load_solution
from pca_ensemble.reactors import simulate_ignition


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = load_yaml(ROOT / "config" / "smoke.yaml")
    mechanism = config["mechanism"]
    case = config["case"]
    audit = audit_mechanism(mechanism["path"], mechanism.get("phase"))
    gas = load_solution(mechanism["path"], mechanism.get("phase"))
    result = simulate_ignition(
        gas=gas,
        temperature_K=float(case["temperature_K"]),
        pressure_Pa=float(case["pressure_bar"]) * 1.0e5,
        equivalence_ratio=float(case["equivalence_ratio"]),
        cracking_ratio=float(case["cracking_ratio"]),
        max_time_s=float(case["max_time_s"]),
    )
    payload = {
        "warning": "Smoke-test output only; not experimental or publication evidence.",
        "mechanism_id": mechanism["id"],
        "mechanism_audit": audit.to_dict(),
        "case": case,
        "result": result.to_dict(),
    }
    destination = ROOT / "results" / "logs" / "smoke_ignition.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

