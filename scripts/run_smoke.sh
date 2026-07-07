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
python3 -m limes_workspace_lens export-prompts examples/workspace_audit_spec.json --out "${TMP_DIR}/prompts.jsonl"
python3 -m limes_workspace_lens summarize-readouts examples/synthetic_readouts.json \
  --spec examples/workspace_audit_spec.json \
  --out "${TMP_DIR}/audit-card.md" \
  --json-out "${TMP_DIR}/audit-card.json"
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
for name in ["audit-card.json", "intervention-plan.json", "comparison.json"]:
    json.loads((root / name).read_text(encoding="utf-8"))
assert (root / "reflection.jsonl").read_text(encoding="utf-8").count("\n") == 9
assert "Workspace Lens" in (root / "comparison.md").read_text(encoding="utf-8")
PY
git diff --check
