"""Assemble the evidence-locked Markdown manuscript from section drafts."""

from __future__ import annotations

from pathlib import Path


TITLE = "Study-held-out validation exposes evidence gaps in multi-mechanism screening of partially cracked ammonia"
SECTIONS = [
    "abstract_draft.md",
    "introduction_draft.md",
    "methods_draft.md",
    "results_draft.md",
    "discussion_draft.md",
    "conclusion_draft.md",
    "code_data_availability.md",
    "references_draft.md",
]


def strip_first_heading(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join(lines).strip()


def main() -> None:
    root = Path("manuscript")
    content = [
        f"# {TITLE}\n",
        "**Pre-submission manuscript — author metadata and repository identifiers pending**\n",
    ]
    # Submission output must not contain internal production-status notes.
    content = [f"# {TITLE}\n"]
    for name in SECTIONS:
        path = root / name
        text = path.read_text(encoding="utf-8")
        content.append(text.strip() if name == "references_draft.md" else strip_first_heading(text))
        content.append("")
    output = root / "manuscript_working.md"
    output.write_text("\n\n".join(content).strip() + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
