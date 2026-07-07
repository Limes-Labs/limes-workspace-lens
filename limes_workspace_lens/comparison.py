from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


COMPARISON_SCHEMA = "limes-workspace-lens/comparison.v0.1"


def compare_reports(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_counts = _counts(before)
    after_counts = _counts(after)
    categories = sorted(set(before_counts) | set(after_counts))
    category_deltas = [
        {
            "category": category,
            "before": before_counts.get(category, 0),
            "after": after_counts.get(category, 0),
            "delta": after_counts.get(category, 0) - before_counts.get(category, 0),
        }
        for category in categories
    ]

    prompt_deltas = []
    before_prompts = {row["prompt_id"]: row for row in before.get("prompt_summaries", [])}
    after_prompts = {row["prompt_id"]: row for row in after.get("prompt_summaries", [])}
    for prompt_id in sorted(set(before_prompts) | set(after_prompts)):
        old = before_prompts.get(prompt_id, {})
        new = after_prompts.get(prompt_id, {})
        prompt_deltas.append(
            {
                "prompt_id": prompt_id,
                "before_status": old.get("status", "missing"),
                "after_status": new.get("status", "missing"),
                "expected_delta": new.get("expected_workspace_term_hits", 0)
                - old.get("expected_workspace_term_hits", 0),
                "audit_delta": new.get("audit_term_hits", 0) - old.get("audit_term_hits", 0),
            }
        )

    return {
        "schema_version": COMPARISON_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "before": _identity(before),
        "after": _identity(after),
        "category_deltas": category_deltas,
        "prompt_deltas": prompt_deltas,
        "interpretation": [
            "Deltas are diagnostic signals, not training-quality scores.",
            "Treat increases in safety or deception vocabularies as prompts for review, not as proof of improvement or harm.",
            "Compare only reports produced with compatible model, tokenizer, lens-fit, layer, position, and top-k settings.",
        ],
    }


def render_markdown_comparison(comparison: dict[str, Any]) -> str:
    before = comparison.get("before", {})
    after = comparison.get("after", {})
    lines = [
        "# Workspace Lens Checkpoint Comparison",
        "",
        "## Scope",
        "",
        f"- Before: `{before.get('label', 'before')}`",
        f"- After: `{after.get('label', 'after')}`",
        "",
        "## Category Deltas",
        "",
        "| Category | Before | After | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in comparison.get("category_deltas", []):
        lines.append(f"| `{row['category']}` | {row['before']} | {row['after']} | {row['delta']} |")

    lines.extend(
        [
            "",
            "## Prompt Deltas",
            "",
            "| Prompt | Before | After | Expected-term Delta | Audit-term Delta |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for row in comparison.get("prompt_deltas", []):
        lines.append(
            f"| `{row['prompt_id']}` | `{row['before_status']}` | `{row['after_status']}` | {row['expected_delta']} | {row['audit_delta']} |"
        )

    lines.extend(["", "## Interpretation", ""])
    for item in comparison.get("interpretation", []):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _counts(report: dict[str, Any]) -> dict[str, int]:
    return {key: int(value) for key, value in report.get("category_counts", {}).items()}


def _identity(report: dict[str, Any]) -> dict[str, Any]:
    project = report.get("project", {})
    model = report.get("model", {})
    lens = report.get("lens", {})
    readouts = report.get("input_readouts", {})
    return {
        "label": project.get("name") or readouts.get("source") or model.get("name") or "unknown",
        "model": model.get("name", "unknown"),
        "lens": lens.get("source", "unknown"),
        "readout_source": readouts.get("source", "unknown"),
    }
