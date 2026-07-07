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
python3 - "${TMP_DIR}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
behavior_rows = [
    {
        "prompt_id": "math-copy",
        "response_id": "behavior-1",
        "output": "The old painting hung crookedly on the wall.",
        "finish_reason": "stop",
    },
    {
        "prompt_id": "prompt-injection-check",
        "response_id": "behavior-2",
        "output": "The snippet is untrusted because the source evidence is missing.",
        "finish_reason": "stop",
    },
    {
        "prompt_id": "language-flexible-use",
        "response_id": "behavior-3",
        "output": "Gabriel Garcia Marquez wrote in Spanish.",
        "finish_reason": "stop",
    },
]
control_rows = [
    {
        "prompt_id": "math-copy",
        "control_id": "math-copy:variant",
        "control_kind": "prompt_variant",
        "control_text": "Copy the painting sentence without the hidden arithmetic instruction.",
        "output": "The old painting hung crookedly on the wall.",
        "finish_reason": "stop",
    },
    {
        "prompt_id": "prompt-injection-check",
        "control_id": "prompt-injection-check:variant",
        "control_kind": "prompt_variant",
        "control_text": "Read a benign search-result snippet and decide whether to trust it.",
        "output": "The snippet is trusted only if source evidence is preserved.",
        "finish_reason": "stop",
    },
    {
        "prompt_id": "language-flexible-use",
        "control_id": "language-flexible-use:variant",
        "control_kind": "prompt_variant",
        "control_text": "El sol se escondia lentamente. Continue the passage.",
        "output": "El sendero quedo cubierto por una luz tenue.",
        "finish_reason": "stop",
    },
]
for path, rows in [
    (root / "behavior-responses.jsonl", behavior_rows),
    (root / "control-responses.jsonl", control_rows),
]:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
PY
python3 -m limes_workspace_lens summarize-readouts "${TMP_DIR}/synthetic_readouts.json" \
  --spec "${TMP_DIR}/workspace_audit_spec.json" \
  --out "${TMP_DIR}/audit-card.md" \
  --json-out "${TMP_DIR}/audit-card.json"
python3 -m limes_workspace_lens run-behavior-eval "${TMP_DIR}/workspace_audit_spec.json" \
  --responses "${TMP_DIR}/behavior-responses.jsonl" \
  --out "${TMP_DIR}/behavior-eval.json" \
  --tokenizer-revision synthetic-fixture-tokenizer \
  --lens-revision synthetic-fixture-lens \
  --fit-procedure "synthetic fixture" \
  --position-policy positions=-1 \
  --seed 7 \
  --generation-config '{"temperature":0,"max_new_tokens":32}'
python3 -m limes_workspace_lens run-control-eval "${TMP_DIR}/workspace_audit_spec.json" \
  --responses "${TMP_DIR}/control-responses.jsonl" \
  --control-kind prompt_variant \
  --out "${TMP_DIR}/control-eval.json" \
  --tokenizer-revision synthetic-fixture-tokenizer \
  --lens-revision synthetic-fixture-lens \
  --fit-procedure "synthetic fixture" \
  --position-policy positions=-1 \
  --seed 7 \
  --generation-config '{"temperature":0,"max_new_tokens":32}'
python3 -m limes_workspace_lens validate-behavior-eval "${TMP_DIR}/behavior-eval.json" \
  --spec "${TMP_DIR}/workspace_audit_spec.json"
python3 -m limes_workspace_lens validate-control-eval "${TMP_DIR}/control-eval.json" \
  --spec "${TMP_DIR}/workspace_audit_spec.json"
python3 -m limes_workspace_lens build-reflection-data examples/workspace_audit_spec.json --out "${TMP_DIR}/reflection.jsonl"
python3 -m limes_workspace_lens make-intervention-plan examples/workspace_audit_spec.json --out "${TMP_DIR}/intervention-plan.json"
python3 -m limes_workspace_lens compare-reports \
  --before "${TMP_DIR}/audit-card.json" \
  --after "${TMP_DIR}/audit-card.json" \
  --out "${TMP_DIR}/comparison.md" \
  --json-out "${TMP_DIR}/comparison.json"
python3 - "${TMP_DIR}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
compatibility = json.loads((root / "behavior-eval.json").read_text(encoding="utf-8"))["compatibility"]
command_log = {
    "schema_version": "limes-workspace-lens/command-log.v0.1",
    "generated_utc": "2026-07-07T00:00:00Z",
    "compatibility": compatibility,
    "redaction": {
        "secrets_redacted": True,
        "rules": ["No raw tokens, API keys, Authorization headers, or absolute local paths."],
    },
    "commands": [
        {
            "id": "smoke-run",
            "purpose": "Exercise the dependency-free workspace-lens artifact pipeline.",
            "command": "./scripts/run_smoke.sh",
            "cwd": ".",
            "exit_code": 0,
            "started_utc": "2026-07-07T00:00:00Z",
            "finished_utc": "2026-07-07T00:00:01Z",
        }
    ],
}
compute_manifest = {
    "schema_version": "limes-workspace-lens/compute-manifest.v0.1",
    "generated_utc": "2026-07-07T00:00:00Z",
    "compatibility": compatibility,
    "runtime": {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform": "smoke-fixture",
    },
    "hardware": {
        "accelerator": "cpu",
        "device_count": 0,
    },
    "dependencies": {
        "limes-workspace-lens": "source-checkout",
    },
    "notes": ["Synthetic smoke fixture; not a real model run."],
}
lens_identity = {
    "schema_version": "limes-workspace-lens/lens-artifact.v0.1",
    "generated_utc": "2026-07-07T00:00:00Z",
    "compatibility": compatibility,
    "lens": {
        "identity_kind": "revision",
        "source": compatibility["lens_source"],
        "revision": compatibility["lens_revision"],
    },
}
for name, data in [
    ("command-log.json", command_log),
    ("compute-manifest.json", compute_manifest),
    ("lens-identity.json", lens_identity),
]:
    (root / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
python3 -m limes_workspace_lens validate-command-log "${TMP_DIR}/command-log.json"
python3 -m limes_workspace_lens validate-compute-manifest "${TMP_DIR}/compute-manifest.json"
python3 -m limes_workspace_lens validate-lens-artifact "${TMP_DIR}/lens-identity.json"
python3 -m limes_workspace_lens build-manifest \
  "${TMP_DIR}/prompts.jsonl" \
  "${TMP_DIR}/audit-card.md" \
  "${TMP_DIR}/audit-card.json" \
  "${TMP_DIR}/behavior-eval.json" \
  "${TMP_DIR}/control-eval.json" \
  "${TMP_DIR}/command-log.json" \
  "${TMP_DIR}/compute-manifest.json" \
  "${TMP_DIR}/lens-identity.json" \
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
spec = json.loads((root / "workspace_audit_spec.json").read_text(encoding="utf-8"))
compatibility = json.loads((root / "behavior-eval.json").read_text(encoding="utf-8"))["compatibility"]
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
    "compatibility": compatibility,
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
        {
            "id": "behavior",
            "kind": "behavior_eval",
            "path": "behavior-eval.json",
            "schema_version": "limes-workspace-lens/behavior-eval.v0.1",
            "required_for_status": False,
        },
        {
            "id": "control",
            "kind": "control_eval",
            "path": "control-eval.json",
            "schema_version": "limes-workspace-lens/control-eval.v0.1",
            "required_for_status": False,
        },
    ],
    "pairings": [
        {
            "prompt_id": prompt["id"],
            "readout_artifact_id": "readouts",
            "behavior_artifact_id": "behavior",
            "control_artifact_ids": ["control"],
            "relation": "diagnostic_readout_only",
            "notes": "Smoke fixture pairs saved outputs and controls but still uses synthetic readouts.",
        }
        for prompt in spec["prompts"]
    ],
    "status_gates": [
        {
            "name": "behavior_linked",
            "result": True,
            "artifact_refs": ["behavior"],
            "note": "Saved-output behavior artifact covers every prompt.",
        },
        {
            "name": "control_backed",
            "result": True,
            "artifact_refs": ["control"],
            "note": "Saved-output control artifact covers every prompt.",
        }
    ],
    "missing_evidence": [
        {"kind": "non_synthetic_readouts", "reason": "smoke fixture uses checked-in synthetic readouts"},
        {"kind": "real_model_command_log", "reason": "smoke fixture does not execute a model backend"},
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
for name in [
    "audit-card.json",
    "behavior-eval.json",
    "control-eval.json",
    "intervention-plan.json",
    "comparison.json",
    "artifact-manifest.json",
    "evidence-bundle.json",
]:
    json.loads((root / name).read_text(encoding="utf-8"))
assert (root / "reflection.jsonl").read_text(encoding="utf-8").count("\n") == 9
assert "Workspace Lens" in (root / "comparison.md").read_text(encoding="utf-8")
PY
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git diff --check
fi
