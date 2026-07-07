#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

cd "${ROOT_DIR}"

python3 -m unittest discover -s tests
python3 -m py_compile limes_workspace_lens/*.py scripts/*.py
python3 -m limes_workspace_lens validate-spec examples/workspace_audit_spec.json
python3 -m limes_workspace_lens validate-readouts examples/synthetic_readouts.json
python3 -m limes_workspace_lens --help >/dev/null
python3 scripts/fit_jlens.py --help >/dev/null
python3 scripts/export_jlens_readouts.py --help >/dev/null
cp examples/workspace_audit_spec.json "${TMP_DIR}/workspace_audit_spec.json"
cp examples/synthetic_readouts.json "${TMP_DIR}/synthetic_readouts.json"
python3 -m limes_workspace_lens export-prompts examples/workspace_audit_spec.json --out "${TMP_DIR}/prompts.jsonl"
python3 -m limes_workspace_lens summarize-readouts "${TMP_DIR}/synthetic_readouts.json" \
  --spec "${TMP_DIR}/workspace_audit_spec.json" \
  --out "${TMP_DIR}/audit-card.md" \
  --json-out "${TMP_DIR}/audit-card.json"
python3 -m limes_workspace_lens build-reflection-data examples/workspace_audit_spec.json --out "${TMP_DIR}/reflection.jsonl"
python3 -m limes_workspace_lens make-intervention-plan examples/workspace_audit_spec.json --out "${TMP_DIR}/intervention-plan.json"
python3 -m limes_workspace_lens compare-reports \
  --before "${TMP_DIR}/audit-card.json" \
  --after "${TMP_DIR}/audit-card.json" \
  --out "${TMP_DIR}/comparison.md" \
  --json-out "${TMP_DIR}/comparison.json"
python3 -m limes_workspace_lens build-manifest \
  "${TMP_DIR}/prompts.jsonl" \
  "${TMP_DIR}/audit-card.md" \
  "${TMP_DIR}/audit-card.json" \
  "${TMP_DIR}/reflection.jsonl" \
  "${TMP_DIR}/intervention-plan.json" \
  "${TMP_DIR}/comparison.md" \
  "${TMP_DIR}/comparison.json" \
  --root "${TMP_DIR}" \
  --out "${TMP_DIR}/artifact-manifest.json" \
  --command "./scripts/run_smoke.sh" \
  --metadata evidence_status=synthetic-fixture
python3 -m limes_workspace_lens validate-manifest "${TMP_DIR}/artifact-manifest.json" --root "${TMP_DIR}"
python3 - "${TMP_DIR}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
bundle = {
    "schema_version": "limes-workspace-lens/evidence-bundle.v0.1",
    "bundle_id": "synthetic-smoke-diagnostic",
    "status": "diagnostic",
    "claim": {
        "question": "Does the smoke fixture exercise the workspace-lens artifact pipeline?",
        "hypothesis": "The synthetic fixture should validate only the artifact plumbing.",
        "interpretation": "This is a diagnostic CI fixture, not a model claim.",
        "claim_scope": "hypothesis_generation",
        "non_claims": ["not a real model result", "not a behavior claim"],
    },
    "compatibility": {
        "model_checkpoint": "replace-with-huggingface-or-local-checkpoint",
        "tokenizer_revision": "synthetic-fixture-tokenizer",
        "lens_source": "anthropics/jacobian-lens or compatible implementation",
        "lens_revision": "synthetic-fixture-lens",
        "prompt_suite_hash": "synthetic-fixture-prompts",
        "top_k": 10,
        "layer_policy": "workspace_layer_range=24-40",
        "position_policy": "positions=-1",
        "fit_procedure": "synthetic fixture",
    },
    "artifacts": [
        {
            "id": "spec",
            "kind": "audit_spec",
            "path": "workspace_audit_spec.json",
            "schema_version": "limes-workspace-lens/audit-spec.v0.1",
            "required_for_status": True,
        },
        {
            "id": "readouts",
            "kind": "readouts",
            "path": "synthetic_readouts.json",
            "schema_version": "limes-workspace-lens/readouts.v0.1",
            "required_for_status": True,
        },
        {
            "id": "report",
            "kind": "audit_report_json",
            "path": "audit-card.json",
            "schema_version": "limes-workspace-lens/report.v0.1",
            "required_for_status": True,
        },
    ],
    "pairings": [
        {
            "prompt_id": "math-copy",
            "readout_artifact_id": "readouts",
            "control_artifact_ids": [],
            "relation": "diagnostic_readout_only",
            "notes": "Smoke fixture intentionally lacks behavior and controls.",
        }
    ],
    "status_gates": [
        {
            "name": "behavior_linked",
            "result": False,
            "artifact_refs": ["spec", "readouts", "report"],
            "note": "Behavior artifact is intentionally absent in the synthetic smoke fixture.",
        }
    ],
    "missing_evidence": [
        {"kind": "behavior_eval", "reason": "not run for synthetic smoke fixture"},
        {"kind": "control_eval", "reason": "not run for synthetic smoke fixture"},
    ],
    "conflicting_findings": [],
    "review": {
        "reviewer": "smoke-test",
        "reviewed_utc": "2026-07-07T00:00:00Z",
        "limitations": ["Synthetic fixture only."],
        "promotion_recommendation": "hold",
    },
}
(root / "evidence-bundle.json").write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
python3 -m limes_workspace_lens validate-bundle "${TMP_DIR}/evidence-bundle.json" \
  --root "${TMP_DIR}" \
  --strict \
  --expected-status diagnostic
python3 - "${TMP_DIR}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for name in ["audit-card.json", "intervention-plan.json", "comparison.json", "artifact-manifest.json", "evidence-bundle.json"]:
    json.loads((root / name).read_text(encoding="utf-8"))
assert (root / "reflection.jsonl").read_text(encoding="utf-8").count("\n") == 9
assert "Workspace Lens" in (root / "comparison.md").read_text(encoding="utf-8")
PY
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git diff --check
fi
