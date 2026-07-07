#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

BUILD_VENV="${TMP_DIR}/build-venv"
DIST_DIR="${TMP_DIR}/dist"
WHEEL_VENV="${TMP_DIR}/wheel-venv"
SDIST_VENV="${TMP_DIR}/sdist-venv"
mkdir -p "${DIST_DIR}"

python3 -m venv "${BUILD_VENV}"
"${BUILD_VENV}/bin/python" -m pip install --upgrade pip build
"${BUILD_VENV}/bin/python" -m build --sdist --wheel --outdir "${DIST_DIR}" "${ROOT_DIR}"

WHEEL_PATH="$(find "${DIST_DIR}" -maxdepth 1 -name '*.whl' -print -quit)"
SDIST_PATH="$(find "${DIST_DIR}" -maxdepth 1 -name '*.tar.gz' -print -quit)"
if [[ -z "${WHEEL_PATH}" || -z "${SDIST_PATH}" ]]; then
  echo "expected both wheel and sdist artifacts in ${DIST_DIR}" >&2
  exit 1
fi

install_and_smoke() {
  local venv_dir="$1"
  local package_path="$2"
  local label="$3"
  local run_dir="${TMP_DIR}/${label}-run"

  python3 -m venv "${venv_dir}"
  "${venv_dir}/bin/python" -m pip install --upgrade pip
  "${venv_dir}/bin/python" -m pip install "${package_path}"
  "${venv_dir}/bin/python" -m pip check

  mkdir -p "${run_dir}"
  cd "${run_dir}"
  "${venv_dir}/bin/limes-workspace-lens" --help >/dev/null
  "${venv_dir}/bin/limes-workspace-lens" init-spec --out spec.json
  "${venv_dir}/bin/limes-workspace-lens" validate-spec spec.json
  "${venv_dir}/bin/limes-workspace-lens" export-prompts spec.json --out prompts.jsonl

  "${venv_dir}/bin/python" - <<'PY'
import json
import pathlib

root = pathlib.Path(".")
spec = json.loads((root / "spec.json").read_text(encoding="utf-8"))
readouts = {
    "schema_version": "limes-workspace-lens/readouts.v0.1",
    "source": "installed-package-smoke",
    "synthetic": True,
    "top_k": spec["lens"]["top_k"],
    "readouts": [],
}
behavior_rows = []
control_rows = []
for prompt in spec["prompts"]:
    prompt_id = prompt["id"]
    expected_terms = prompt.get("expected_workspace_terms") or ["evidence"]
    readouts["readouts"].append(
        {
            "prompt_id": prompt_id,
            "position": -1,
            "layer": spec["lens"]["workspace_layer_range"][0],
            "top_tokens": [
                {"token": str(expected_terms[0]), "rank": 1, "score": 1.0},
                {"token": "evidence", "rank": 2, "score": 0.5},
            ],
        }
    )
    behavior_rows.append(
        {
            "prompt_id": prompt_id,
            "response_id": f"{prompt_id}:behavior",
            "output": "Synthetic installed-package output with no model execution.",
            "finish_reason": "stop",
        }
    )
    control_rows.append(
        {
            "prompt_id": prompt_id,
            "control_id": f"{prompt_id}:prompt-variant",
            "control_kind": "prompt_variant",
            "control_text": f"Prompt variant for {prompt_id}.",
            "output": "Synthetic installed-package control output.",
            "finish_reason": "stop",
        }
    )

(root / "readouts.json").write_text(json.dumps(readouts, indent=2, sort_keys=True) + "\n", encoding="utf-8")
for path, rows in [
    (root / "behavior-responses.jsonl", behavior_rows),
    (root / "control-responses.jsonl", control_rows),
]:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
PY

  "${venv_dir}/bin/limes-workspace-lens" validate-readouts readouts.json
  "${venv_dir}/bin/limes-workspace-lens" summarize-readouts readouts.json \
    --spec spec.json \
    --out audit-card.md \
    --json-out audit-card.json
  "${venv_dir}/bin/limes-workspace-lens" run-behavior-eval spec.json \
    --responses behavior-responses.jsonl \
    --out behavior-eval.json \
    --tokenizer-revision installed-smoke-tokenizer \
    --lens-revision installed-smoke-lens \
    --fit-procedure "installed package smoke" \
    --position-policy positions=-1 \
    --seed 7
  "${venv_dir}/bin/limes-workspace-lens" run-control-eval spec.json \
    --responses control-responses.jsonl \
    --control-kind prompt_variant \
    --out control-eval.json \
    --tokenizer-revision installed-smoke-tokenizer \
    --lens-revision installed-smoke-lens \
    --fit-procedure "installed package smoke" \
    --position-policy positions=-1 \
    --seed 7
  "${venv_dir}/bin/limes-workspace-lens" validate-behavior-eval behavior-eval.json --spec spec.json
  "${venv_dir}/bin/limes-workspace-lens" validate-control-eval control-eval.json --spec spec.json
  "${venv_dir}/bin/limes-workspace-lens" build-reflection-data spec.json --out reflection.jsonl
  "${venv_dir}/bin/limes-workspace-lens" make-intervention-plan spec.json --out intervention-plan.json
  "${venv_dir}/bin/limes-workspace-lens" build-manifest \
    prompts.jsonl \
    audit-card.md \
    audit-card.json \
    behavior-eval.json \
    control-eval.json \
    reflection.jsonl \
    intervention-plan.json \
    --root . \
    --out artifact-manifest.json \
    --command "scripts/check_install.sh ${label}" \
    --metadata evidence_status=installed-package-smoke
  "${venv_dir}/bin/limes-workspace-lens" validate-manifest artifact-manifest.json --root .

  "${venv_dir}/bin/python" - <<'PY'
import json
import pathlib

root = pathlib.Path(".")
for name in [
    "spec.json",
    "readouts.json",
    "audit-card.json",
    "behavior-eval.json",
    "control-eval.json",
    "intervention-plan.json",
    "artifact-manifest.json",
]:
    json.loads((root / name).read_text(encoding="utf-8"))
assert (root / "prompts.jsonl").read_text(encoding="utf-8").count("\n") == 3
assert (root / "reflection.jsonl").read_text(encoding="utf-8").count("\n") == 9
report_text = (root / "audit-card.md").read_text(encoding="utf-8")
assert "Audit Categories" in report_text
assert "Prompt Summaries" in report_text
PY
}

install_and_smoke "${WHEEL_VENV}" "${WHEEL_PATH}" "wheel"
install_and_smoke "${SDIST_VENV}" "${SDIST_PATH}" "sdist"
