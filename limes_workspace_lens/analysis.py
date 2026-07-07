from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

from .schema import REPORT_SCHEMA


def normalize_token(token: str) -> str:
    return token.strip().lower()


def score_readouts(
    spec: dict[str, Any],
    readouts: dict[str, Any],
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    audit_terms = {
        category: {normalize_token(term) for term in terms}
        for category, terms in spec.get("audit_terms", {}).items()
    }
    prompt_expected = {
        prompt["id"]: {normalize_token(term) for term in prompt.get("expected_workspace_terms", [])}
        for prompt in spec.get("prompts", [])
    }

    category_counts: Counter[str] = Counter()
    expected_counts: Counter[str] = Counter()
    layer_counts: Counter[int] = Counter()
    prompt_counts: Counter[str] = Counter()
    hits: list[dict[str, Any]] = []

    for row in readouts.get("readouts", []):
        prompt_id = row["prompt_id"]
        layer = row["layer"]
        tokens = row.get("top_tokens", [])[:top_k]
        normalized = [normalize_token(token["token"]) for token in tokens]
        ranks = {
            normalize_token(token["token"]): token.get("rank")
            for token in tokens
            if isinstance(token.get("token"), str)
        }

        matched_categories: list[str] = []
        for category, terms in audit_terms.items():
            matched_terms = sorted(set(normalized) & terms)
            if matched_terms:
                matched_categories.append(category)
                category_counts[category] += len(matched_terms)
                layer_counts[layer] += len(matched_terms)
                prompt_counts[prompt_id] += len(matched_terms)
                for term in matched_terms:
                    hits.append(
                        {
                            "prompt_id": prompt_id,
                            "position": row["position"],
                            "layer": layer,
                            "category": category,
                            "term": term,
                            "rank": ranks.get(term),
                        }
                    )

        expected_terms = prompt_expected.get(prompt_id, set())
        expected_matched = sorted(set(normalized) & expected_terms)
        if expected_matched:
            expected_counts[prompt_id] += len(expected_matched)
            for term in expected_matched:
                hits.append(
                    {
                        "prompt_id": prompt_id,
                        "position": row["position"],
                        "layer": layer,
                        "category": "expected_workspace_term",
                        "term": term,
                        "rank": ranks.get(term),
                    }
                )

    prompt_rows = _prompt_summaries(spec, expected_counts, prompt_counts)
    layer_rows = [
        {"layer": layer, "audit_term_hits": count}
        for layer, count in sorted(layer_counts.items())
    ]

    return {
        "schema_version": REPORT_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "project": spec.get("project", {}),
        "model": spec.get("model", {}),
        "lens": spec.get("lens", {}),
        "input_readouts": {
            "source": readouts.get("source", "unknown"),
            "synthetic": bool(readouts.get("synthetic", False)),
            "row_count": len(readouts.get("readouts", [])),
        },
        "top_k": top_k,
        "category_counts": dict(sorted(category_counts.items())),
        "prompt_summaries": prompt_rows,
        "layer_summaries": layer_rows,
        "hits": hits,
        "interpretation": _interpretation(readouts, category_counts, expected_counts),
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    project = report.get("project", {})
    model = report.get("model", {})
    lens = report.get("lens", {})
    source = report.get("input_readouts", {}).get("source", "unknown")
    synthetic = report.get("input_readouts", {}).get("synthetic", False)

    lines = [
        f"# {project.get('name', 'Workspace Lens Audit')} Report",
        "",
        "## Scope",
        "",
        f"- Model: `{model.get('name', 'unknown')}`",
        f"- Lens source: `{lens.get('source', 'unknown')}`",
        f"- Readout artifact: `{source}`",
        f"- Top-k window: `{report.get('top_k')}`",
    ]
    if synthetic:
        lines.extend(
            [
                "- Evidence status: `synthetic fixture`",
                "",
                "This report is generated from a checked-in fixture. It validates the audit pipeline, not a model behavior claim.",
            ]
        )
    lines.extend(["", "## Audit Categories", ""])
    if report.get("category_counts"):
        lines.extend(["| Category | Hits |", "| --- | ---: |"])
        for category, count in report["category_counts"].items():
            lines.append(f"| `{category}` | {count} |")
    else:
        lines.append("No configured audit terms appeared in the selected top-k readouts.")

    lines.extend(["", "## Prompt Summaries", ""])
    lines.extend(["| Prompt | Expected-term hits | Audit-term hits | Status |", "| --- | ---: | ---: | --- |"])
    for row in report.get("prompt_summaries", []):
        lines.append(
            f"| `{row['prompt_id']}` | {row['expected_workspace_term_hits']} | {row['audit_term_hits']} | `{row['status']}` |"
        )

    lines.extend(["", "## Layer Summary", ""])
    if report.get("layer_summaries"):
        lines.extend(["| Layer | Audit-term hits |", "| ---: | ---: |"])
        for row in report["layer_summaries"]:
            lines.append(f"| {row['layer']} | {row['audit_term_hits']} |")
    else:
        lines.append("No layer-level audit hits were recorded.")

    lines.extend(["", "## Top Hits", ""])
    if report.get("hits"):
        lines.extend(["| Prompt | Position | Layer | Category | Term | Rank |", "| --- | --- | ---: | --- | --- | ---: |"])
        for hit in report["hits"][:50]:
            lines.append(
                f"| `{hit['prompt_id']}` | `{hit['position']}` | {hit['layer']} | `{hit['category']}` | `{hit['term']}` | {hit.get('rank', '')} |"
            )
    else:
        lines.append("No token hits matched the configured vocabulary.")

    lines.extend(["", "## Interpretation", ""])
    for item in report.get("interpretation", []):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _prompt_summaries(
    spec: dict[str, Any],
    expected_counts: Counter[str],
    prompt_counts: Counter[str],
) -> list[dict[str, Any]]:
    rows = []
    for prompt in spec.get("prompts", []):
        prompt_id = prompt["id"]
        expected_hits = expected_counts[prompt_id]
        audit_hits = prompt_counts[prompt_id]
        if expected_hits and audit_hits:
            status = "expected-and-audit-hits"
        elif expected_hits:
            status = "expected-hit"
        elif audit_hits:
            status = "audit-hit"
        else:
            status = "no-hit"
        rows.append(
            {
                "prompt_id": prompt_id,
                "kind": prompt.get("kind"),
                "expected_workspace_term_hits": expected_hits,
                "audit_term_hits": audit_hits,
                "status": status,
            }
        )
    return rows


def _interpretation(
    readouts: dict[str, Any],
    category_counts: Counter[str],
    expected_counts: Counter[str],
) -> list[str]:
    notes = []
    if readouts.get("synthetic"):
        notes.append("Synthetic fixtures are useful for CI only; replace them with model-internal readouts before making behavioral claims.")
    if category_counts:
        notes.append("Matched audit terms should be treated as hypothesis-generation signals until validated with behavior tests and controls.")
    if expected_counts:
        notes.append("Expected workspace terms appeared in at least one prompt, showing that the parser and scoring window are wired correctly.")
    if not category_counts and not expected_counts:
        notes.append("No configured terms appeared; widen the layer/position window or revisit the vocabulary before interpreting this as absence.")
    return notes
