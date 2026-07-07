from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from .schema import REFLECTION_SCHEMA


def build_reflection_rows(spec: dict[str, Any]) -> list[dict[str, Any]]:
    reflection = spec.get("reflection_training", {})
    principles = reflection.get("principles", [])
    if not principles:
        return []

    rows: list[dict[str, Any]] = []
    for prompt in spec.get("prompts", []):
        if prompt.get("exclude_from_reflection_training", False):
            continue
        for principle in principles:
            rows.append(
                {
                    "schema_version": REFLECTION_SCHEMA,
                    "id": f"{prompt['id']}::{_slug(principle)}",
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "source_prompt_id": prompt["id"],
                    "kind": "counterfactual_reflection",
                    "intended_use": "SFT or preference-data candidate for interrupted-reflection continuations",
                    "not_a_behavior_label": True,
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                f"{prompt['text']}\n\n"
                                "Counterfactual interruption: before taking the action, state the principle "
                                "that should be active in the model's internal workspace."
                            ),
                        },
                        {
                            "role": "assistant",
                            "content": f"Relevant principle: {principle}",
                        },
                    ],
                }
            )
    return rows


def _slug(text: str) -> str:
    lowered = text.strip().lower()
    return re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")[:80] or "principle"
