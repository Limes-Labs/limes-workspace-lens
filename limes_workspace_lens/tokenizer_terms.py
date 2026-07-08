from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .jlens_adapter import AdapterError, public_identifier, sha256_file
from .schema import TOKENIZER_TERM_MAP_SCHEMA, public_artifact_path_label


@dataclass(frozen=True)
class TermMatcher:
    term: str
    normalized: str
    token_ids: frozenset[int]
    normalized_variants: frozenset[str]

    def match(self, *, token: str, token_id: int | None) -> str | None:
        if token_id is not None and token_id in self.token_ids:
            return "token_id"
        if normalize_text(token) in self.normalized_variants:
            return "normalized_token"
        return None


def normalize_text(value: str) -> str:
    return value.strip().casefold()


def collect_spec_terms(spec: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category, terms in spec.get("audit_terms", {}).items():
        if not isinstance(category, str) or not isinstance(terms, list):
            continue
        for term in terms:
            if isinstance(term, str) and term.strip():
                rows.append(
                    {
                        "scope": "audit_terms",
                        "category": category,
                        "term": term,
                    }
                )
    for prompt in spec.get("prompts", []):
        if not isinstance(prompt, dict):
            continue
        prompt_id = prompt.get("id")
        if not isinstance(prompt_id, str) or not prompt_id:
            continue
        for term in prompt.get("expected_workspace_terms", []):
            if isinstance(term, str) and term.strip():
                rows.append(
                    {
                        "scope": "expected_workspace_terms",
                        "prompt_id": prompt_id,
                        "term": term,
                    }
                )
    return rows


def term_variants(term: str) -> list[dict[str, Any]]:
    stripped = term.strip()
    folded = stripped.casefold()
    candidates = [
        ("raw", stripped),
        ("leading_space", f" {stripped}"),
    ]
    if folded != stripped:
        candidates.extend(
            [
                ("casefold", folded),
                ("leading_space_casefold", f" {folded}"),
            ]
        )
    seen: set[str] = set()
    variants: list[dict[str, Any]] = []
    for kind, text in candidates:
        if text and text not in seen:
            variants.append({"kind": kind, "text": text, "normalized": normalize_text(text)})
            seen.add(text)
    return variants


def build_tokenizer_term_map(
    spec: dict[str, Any],
    *,
    tokenizer: Any,
    model: str,
    tokenizer_revision: str | None,
    spec_path: str | Path | None,
    local_files_only: bool,
    trust_remote_code: bool,
    synthetic: bool = False,
) -> dict[str, Any]:
    terms: list[dict[str, Any]] = []
    for row in collect_spec_terms(spec):
        term = row["term"]
        mapped_variants = []
        for variant in term_variants(term):
            token_ids = _encode_variant(tokenizer, variant["text"])
            mapped_variants.append(
                {
                    **variant,
                    "token_ids": token_ids,
                    "tokens": _tokens_for_ids(tokenizer, token_ids),
                    "single_token": len(token_ids) == 1,
                }
            )
        terms.append(
            {
                **row,
                "normalized": normalize_text(term),
                "variants": mapped_variants,
                "single_token_token_ids": sorted(
                    {
                        token_id
                        for variant in mapped_variants
                        if variant["single_token"]
                        for token_id in variant["token_ids"]
                    }
                ),
                "multi_token_variant_count": sum(
                    1 for variant in mapped_variants if not variant["single_token"]
                ),
            }
        )
    spec_file = Path(spec_path) if spec_path is not None else None
    return {
        "schema_version": TOKENIZER_TERM_MAP_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": f"hf-tokenizer-term-map:{public_identifier(model)}",
        "synthetic": synthetic,
        "tokenizer": {
            "id": public_identifier(model),
            "revision": tokenizer_revision,
        },
        "input_spec": {
            "path": public_artifact_path_label(spec_file) if spec_file else None,
            "sha256": sha256_file(spec_file) if spec_file else None,
        },
        "normalization": {
            "casefold": True,
            "strip": True,
            "variant_policy": ["raw", "leading_space", "casefold_when_changed"],
        },
        "generation": {
            "adapter_version": __version__,
            "dependency_profile": "transformers-tokenizer",
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
        },
        "terms": terms,
    }


def build_match_index(term_map: dict[str, Any]) -> dict[str, dict[str, list[TermMatcher]]]:
    index: dict[str, dict[str, list[TermMatcher]]] = {
        "audit_terms": {},
        "expected_workspace_terms": {},
    }
    terms = term_map.get("terms", [])
    if not isinstance(terms, list):
        return index
    for row in terms:
        if not isinstance(row, dict):
            continue
        scope = row.get("scope")
        if scope not in index:
            continue
        key = row.get("category") if scope == "audit_terms" else row.get("prompt_id")
        term = row.get("term")
        if not isinstance(key, str) or not key or not isinstance(term, str) or not term.strip():
            continue
        single_token_variants = [
            variant
            for variant in row.get("variants", [])
            if isinstance(variant, dict) and variant.get("single_token") is True
        ]
        matcher = TermMatcher(
            term=term,
            normalized=normalize_text(term),
            token_ids=frozenset(
                token_id
                for variant in single_token_variants
                for token_id in variant.get("token_ids", [])
                if isinstance(token_id, int) and not isinstance(token_id, bool)
            ),
            normalized_variants=frozenset(
                normalize_text(variant["text"])
                for variant in single_token_variants
                if isinstance(variant.get("text"), str) and variant.get("text", "").strip()
            ),
        )
        index[scope].setdefault(key, []).append(matcher)
    return index


def _encode_variant(tokenizer: Any, text: str) -> list[int]:
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    except Exception as exc:
        raise AdapterError(f"tokenizer failed to encode variant {text!r}: {exc}") from exc
    if not isinstance(token_ids, list) or not token_ids:
        raise AdapterError(f"tokenizer produced no token ids for variant {text!r}")
    for token_id in token_ids:
        if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
            raise AdapterError(f"tokenizer produced invalid token id {token_id!r} for {text!r}")
    return token_ids


def _tokens_for_ids(tokenizer: Any, token_ids: list[int]) -> list[str]:
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        converted = tokenizer.convert_ids_to_tokens(token_ids)
        if isinstance(converted, list) and all(isinstance(item, str) for item in converted):
            return converted
        if isinstance(converted, str):
            return [converted]
    tokens: list[str] = []
    for token_id in token_ids:
        try:
            tokens.append(str(tokenizer.decode([token_id])))
        except Exception:
            tokens.append(f"<token:{token_id}>")
    return tokens
