from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .schema import INTERVENTION_SCHEMA


def build_intervention_plan(spec: dict[str, Any]) -> dict[str, Any]:
    interventions = []
    for item in spec.get("interventions", []):
        interventions.append(
            {
                "id": item["id"],
                "prompt_id": item["prompt_id"],
                "kind": item["kind"],
                "layer_range": item.get("layer_range", spec.get("lens", {}).get("workspace_layer_range")),
                "positions": item.get("positions", "all_prompt_positions"),
                "source_token": item.get("source_token"),
                "target_token": item.get("target_token"),
                "terms": item.get("terms", []),
                "strength": item.get("strength", 1.0),
                "control": item.get("control", "random_direction_or_noop_control_required"),
                "notes": item.get("notes", ""),
            }
        )
    return {
        "schema_version": INTERVENTION_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "project": spec.get("project", {}),
        "model": spec.get("model", {}),
        "lens": spec.get("lens", {}),
        "warning": (
            "This is an intervention plan, not an executable model-surgery script. "
            "Run only in an internal-access environment with behavior controls."
        ),
        "interventions": interventions,
    }
