from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

from .schema import REPORT_SCHEMA, public_artifact_path_label
from .tokenizer_terms import TermMatcher, build_match_index


def normalize_token(token: str) -> str:
    return token.strip().casefold()


def score_readouts(
    spec: dict[str, Any],
    readouts: dict[str, Any],
    *,
    top_k: int = 10,
    term_map: dict[str, Any] | None = None,
    term_map_path: str | None = None,
    term_map_sha256: str | None = None,
) -> dict[str, Any]:
    match_index = build_match_index(term_map) if term_map is not None else None
    audit_terms = _audit_term_matchers(spec, match_index)
    prompt_expected = _expected_term_matchers(spec, match_index)

    category_counts: Counter[str] = Counter()
    expected_counts: Counter[str] = Counter()
    layer_counts: Counter[int] = Counter()
    prompt_counts: Counter[str] = Counter()
    hits: list[dict[str, Any]] = []

    for row in readouts.get("readouts", []):
        prompt_id = row["prompt_id"]
        layer = row["layer"]
        tokens = row.get("top_tokens", [])[:top_k]
        matched_categories: list[str] = []
        for category, matchers in audit_terms.items():
            matched_terms = _match_terms(tokens, matchers)
            if matched_terms:
                matched_categories.append(category)
                category_counts[category] += len(matched_terms)
                layer_counts[layer] += len(matched_terms)
                prompt_counts[prompt_id] += len(matched_terms)
                for term, match in sorted(matched_terms.items()):
                    hits.append(
                        {
                            "prompt_id": prompt_id,
                            "position": row["position"],
                            "layer": layer,
                            "category": category,
                            "term": term,
                            "rank": match.get("rank"),
                            "match_kind": match.get("match_kind"),
                            "matched_token": match.get("matched_token"),
                            "token_id": match.get("token_id"),
                        }
                    )

        expected_terms = prompt_expected.get(prompt_id, [])
        expected_matched = _match_terms(tokens, expected_terms)
        if expected_matched:
            expected_counts[prompt_id] += len(expected_matched)
            for term, match in sorted(expected_matched.items()):
                hits.append(
                    {
                        "prompt_id": prompt_id,
                        "position": row["position"],
                        "layer": layer,
                        "category": "expected_workspace_term",
                        "term": term,
                        "rank": match.get("rank"),
                        "match_kind": match.get("match_kind"),
                        "matched_token": match.get("matched_token"),
                        "token_id": match.get("token_id"),
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
            "tokenizer_term_map": _term_map_summary(
                term_map,
                path=term_map_path,
                sha256=term_map_sha256,
            ),
        },
        "top_k": top_k,
        "category_counts": dict(sorted(category_counts.items())),
        "prompt_summaries": prompt_rows,
        "layer_summaries": layer_rows,
        "hits": hits,
        "interpretation": _interpretation(readouts, category_counts, expected_counts),
    }


def _audit_term_matchers(
    spec: dict[str, Any],
    match_index: dict[str, dict[str, list[TermMatcher]]] | None,
) -> dict[str, list[TermMatcher]]:
    if match_index is not None:
        return match_index.get("audit_terms", {})
    return {
        category: [
            TermMatcher(
                term=term,
                normalized=normalize_token(term),
                token_ids=frozenset(),
                normalized_variants=frozenset({normalize_token(term)}),
            )
            for term in terms
        ]
        for category, terms in spec.get("audit_terms", {}).items()
    }


def _expected_term_matchers(
    spec: dict[str, Any],
    match_index: dict[str, dict[str, list[TermMatcher]]] | None,
) -> dict[str, list[TermMatcher]]:
    if match_index is not None:
        return match_index.get("expected_workspace_terms", {})
    return {
        prompt["id"]: [
            TermMatcher(
                term=term,
                normalized=normalize_token(term),
                token_ids=frozenset(),
                normalized_variants=frozenset({normalize_token(term)}),
            )
            for term in prompt.get("expected_workspace_terms", [])
        ]
        for prompt in spec.get("prompts", [])
    }


def _match_terms(
    tokens: list[dict[str, Any]],
    matchers: list[TermMatcher],
) -> dict[str, dict[str, Any]]:
    matches: dict[str, dict[str, Any]] = {}
    for token in tokens:
        if not isinstance(token, dict) or not isinstance(token.get("token"), str):
            continue
        token_id = token.get("token_id")
        token_id = token_id if isinstance(token_id, int) and not isinstance(token_id, bool) else None
        token_text = token["token"]
        rank = token.get("rank")
        for matcher in matchers:
            match_kind = matcher.match(token=token_text, token_id=token_id)
            if match_kind is None:
                continue
            current = matches.get(matcher.normalized)
            current_rank = current.get("rank") if isinstance(current, dict) else None
            if current is None or _rank_is_better(rank, current_rank):
                matches[matcher.normalized] = {
                    "term": matcher.term,
                    "rank": rank,
                    "match_kind": match_kind,
                    "matched_token": token_text,
                    "token_id": token_id,
                }
    return matches


def _rank_is_better(rank: Any, current_rank: Any) -> bool:
    if isinstance(rank, int) and not isinstance(rank, bool):
        if isinstance(current_rank, int) and not isinstance(current_rank, bool):
            return rank < current_rank
        return True
    return current_rank is None


def _term_map_summary(
    term_map: dict[str, Any] | None,
    *,
    path: str | None,
    sha256: str | None,
) -> dict[str, Any] | None:
    if term_map is None:
        return None
    tokenizer = term_map.get("tokenizer") if isinstance(term_map.get("tokenizer"), dict) else {}
    return {
        "source": term_map.get("source"),
        "synthetic": term_map.get("synthetic"),
        "path": public_artifact_path_label(path) if path else None,
        "sha256": sha256,
        "tokenizer": {
            "id": tokenizer.get("id"),
            "revision": tokenizer.get("revision"),
        },
        "term_count": len(term_map.get("terms", [])) if isinstance(term_map.get("terms"), list) else 0,
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
