from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


COMPARISON_SCHEMA = "limes-workspace-lens/comparison.v0.1"


def compare_reports(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    compatibility_errors: list[str] | None = None,
    allow_incompatible: bool = False,
) -> dict[str, Any]:
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
        "compatibility": {
            "status": _compatibility_status(compatibility_errors or [], allow_incompatible),
            "errors": compatibility_errors or [],
        },
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
        "## Compatibility",
        "",
        f"- Status: `{comparison.get('compatibility', {}).get('status', 'unknown')}`",
    ]
    compatibility_errors = comparison.get("compatibility", {}).get("errors", [])
    if compatibility_errors:
        lines.extend(["", "| Incompatibility |", "| --- |"])
        for error in compatibility_errors:
            lines.append(f"| {error} |")
    lines.extend(
        [
            "",
        "## Category Deltas",
        "",
        "| Category | Before | After | Delta |",
        "| --- | ---: | ---: | ---: |",
        ]
    )
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


def compatibility_errors(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if before.get("top_k") != after.get("top_k"):
        errors.append(f"top_k differs: before={before.get('top_k')!r}, after={after.get('top_k')!r}")

    before_prompts = {row.get("prompt_id") for row in before.get("prompt_summaries", [])}
    after_prompts = {row.get("prompt_id") for row in after.get("prompt_summaries", [])}
    if before_prompts != after_prompts:
        errors.append(
            "prompt suite differs: "
            f"only_before={sorted(before_prompts - after_prompts)}, "
            f"only_after={sorted(after_prompts - before_prompts)}"
        )

    before_categories = set(before.get("category_counts", {}).keys())
    after_categories = set(after.get("category_counts", {}).keys())
    if before_categories != after_categories:
        errors.append(
            "audit categories differ: "
            f"only_before={sorted(before_categories - after_categories)}, "
            f"only_after={sorted(after_categories - before_categories)}"
        )

    before_lens = before.get("lens", {})
    after_lens = after.get("lens", {})
    for key in ["source", "workspace_layer_range", "top_k"]:
        if before_lens.get(key) != after_lens.get(key):
            errors.append(
                f"lens.{key} differs: before={before_lens.get(key)!r}, after={after_lens.get(key)!r}"
            )

    before_model = before.get("model", {})
    after_model = after.get("model", {})
    if before_model.get("family") != after_model.get("family"):
        errors.append(
            f"model.family differs: before={before_model.get('family')!r}, after={after_model.get('family')!r}"
        )

    before_readouts = before.get("input_readouts", {})
    after_readouts = after.get("input_readouts", {})
    if before_readouts.get("synthetic") != after_readouts.get("synthetic"):
        errors.append(
            "input_readouts.synthetic differs: "
            f"before={before_readouts.get('synthetic')!r}, after={after_readouts.get('synthetic')!r}"
        )
    before_term_map = before_readouts.get("tokenizer_term_map")
    after_term_map = after_readouts.get("tokenizer_term_map")
    before_term_map_identity = _term_map_identity(before_term_map)
    after_term_map_identity = _term_map_identity(after_term_map)
    if before_term_map_identity != after_term_map_identity:
        errors.append(
            "input_readouts.tokenizer_term_map differs: "
            f"before={before_term_map_identity!r}, "
            f"after={after_term_map_identity!r}"
        )
    return errors


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


def _compatibility_status(errors: list[str], allow_incompatible: bool) -> str:
    if not errors:
        return "compatible"
    if allow_incompatible:
        return "incompatible-allowed"
    return "incompatible"


def _term_map_identity(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    tokenizer = value.get("tokenizer") if isinstance(value.get("tokenizer"), dict) else {}
    return {
        "source": value.get("source"),
        "sha256": value.get("sha256"),
        "tokenizer": {
            "id": tokenizer.get("id"),
            "revision": tokenizer.get("revision"),
        },
    }
