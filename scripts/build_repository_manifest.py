"""Build a deterministic repository file manifest for deposition review."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


DEFAULT_ROOTS = (
    "data",
    "figures",
    "literature",
    "manuscript",
    "mechanisms",
    "results",
    "scripts",
    "src",
    "submission",
    "tests",
)
TOP_LEVEL_FILES = ("README.md", "environment.yml", "pyproject.toml")
SKIP_PARTS = {".git", ".pytest_cache", "__pycache__", ".mypy_cache"}
SKIP_NAMES = {"repository_file_manifest.csv"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def disposition(relative: Path) -> tuple[str, str]:
    parts = relative.parts
    suffix = relative.suffix.lower()
    if parts[0] == "mechanisms" and suffix in {".yaml", ".yml", ".ck", ".inp", ".dat"}:
        return "licence_review", "Include only if the third-party licence permits redistribution."
    if parts[:3] == ("results", "logs", "lbv_case_workspace") or parts[:3] == (
        "results",
        "logs",
        "lbv_map_case_workspace",
    ):
        return "optional_raw_diagnostic", "Large per-case workspace; retain failures and compact diagnostics at minimum."
    if parts[0] == "data" and len(parts) > 1 and parts[1] == "raw":
        return "source_rights_review", "Public-source or publisher file; retain provenance and redistribute only when permitted."
    if suffix in {".pyc", ".tmp"}:
        return "exclude", "Generated cache or temporary file."
    return "include", "Author-generated or repository metadata file."


def iter_files(root: Path):
    candidates: list[Path] = []
    for name in DEFAULT_ROOTS:
        path = root / name
        if path.exists():
            candidates.extend(p for p in path.rglob("*") if p.is_file())
    candidates.extend(root / name for name in TOP_LEVEL_FILES if (root / name).is_file())
    for path in sorted(set(candidates), key=lambda p: p.relative_to(root).as_posix().lower()):
        relative = path.relative_to(root)
        if any(part in SKIP_PARTS for part in relative.parts) or path.name in SKIP_NAMES:
            continue
        yield path, relative


def build(root: Path) -> pd.DataFrame:
    rows = []
    for path, relative in iter_files(root):
        status, note = disposition(relative)
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "deposit_status": status,
                "rights_or_packaging_note": note,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output", type=Path, default=Path("submission/repository_file_manifest.csv")
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = build(root)
    frame.to_csv(output, index=False, lineterminator="\n")
    print(f"Wrote {len(frame)} files ({frame['size_bytes'].sum()} bytes) to {output}")
    print(frame.groupby("deposit_status").size().to_string())


if __name__ == "__main__":
    main()
