from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from limes_workspace_lens.analysis import score_readouts
from limes_workspace_lens.schema import (
    load_json,
    validate_audit_spec,
    validate_readouts,
    validate_report,
)


ROOT = Path(__file__).resolve().parents[1]


def spec_fixture() -> dict:
    return load_json(ROOT / "examples" / "workspace_audit_spec.json")


def readout_fixture() -> dict:
    return load_json(ROOT / "examples" / "synthetic_readouts.json")


class NegativeSchemaValidationTests(unittest.TestCase):
    def test_spec_rejects_empty_prompt_fields_and_bad_lens_numbers(self) -> None:
        spec = spec_fixture()
        spec["prompts"][0]["text"] = ""
        spec["prompts"][0]["kind"] = " "
        spec["lens"]["top_k"] = 0
        errors = validate_audit_spec(spec)
        self.assertTrue(any("spec.prompts[0].text" in error for error in errors))
        self.assertTrue(any("spec.prompts[0].kind" in error for error in errors))
        self.assertTrue(any("spec.lens.top_k" in error for error in errors))

    def test_spec_rejects_unknown_intervention_prompt(self) -> None:
        spec = spec_fixture()
        spec["interventions"][0]["prompt_id"] = "missing-prompt"
        errors = validate_audit_spec(spec)
        self.assertTrue(any("unknown prompt id" in error for error in errors))

    def test_spec_rejects_empty_audit_term_category(self) -> None:
        spec = spec_fixture()
        spec["audit_terms"]["empty"] = []
        errors = validate_audit_spec(spec)
        self.assertTrue(any("spec.audit_terms.empty" in error for error in errors))

    def test_readouts_require_source_and_synthetic(self) -> None:
        readouts = readout_fixture()
        readouts.pop("source")
        readouts["synthetic"] = "false"
        errors = validate_readouts(readouts)
        self.assertTrue(any("readouts.source" in error for error in errors))
        self.assertTrue(any("readouts.synthetic" in error for error in errors))

    def test_readouts_reject_unknown_prompt_when_spec_is_supplied(self) -> None:
        spec = spec_fixture()
        readouts = readout_fixture()
        readouts["readouts"][0]["prompt_id"] = "unknown"
        errors = validate_readouts(readouts, spec)
        self.assertTrue(any("unknown prompt id" in error for error in errors))

    def test_readouts_reject_bool_layer_bad_position_rank_zero_and_duplicate_rank(self) -> None:
        readouts = readout_fixture()
        row = copy.deepcopy(readouts["readouts"][0])
        row["layer"] = True
        row["position"] = False
        row["top_tokens"][0]["rank"] = 0
        row["top_tokens"][1]["rank"] = row["top_tokens"][2]["rank"]
        row["top_tokens"][0]["score"] = float("nan")
        readouts["readouts"] = [row]
        errors = validate_readouts(readouts)
        self.assertTrue(any(".layer" in error for error in errors))
        self.assertTrue(any(".position" in error for error in errors))
        self.assertTrue(any(".rank: must be a positive integer" in error for error in errors))
        self.assertTrue(any("duplicate rank" in error for error in errors))
        self.assertTrue(any(".score" in error for error in errors))

    def test_readouts_lint_public_metadata_without_linting_tokens(self) -> None:
        readouts = readout_fixture()
        readouts["readouts"][0]["top_tokens"][0]["token"] = "secret"
        readouts["provenance"] = {"note": "safe public metadata"}
        self.assertEqual([], validate_readouts(readouts))

        readouts["lens_file"] = "../secret/lens.pt"
        readouts["provenance"] = {"lens": {"file": "https://example.com/lens.pt"}}
        path_errors = validate_readouts(readouts)
        self.assertTrue(any("readouts.lens_file" in error for error in path_errors))
        self.assertTrue(any("readouts.provenance.lens.file" in error for error in path_errors))

        readouts["lens_file"] = "lens.pt"
        readouts["provenance"] = {"lens": {"file": "nested/lens.pt"}}
        self.assertEqual([], validate_readouts(readouts))

        readouts["provenance"] = {
            "command": "python /tmp/private/run.py",
            "environment": {"HF_TOKEN": "raw-token-value"},
        }
        errors = validate_readouts(readouts)
        self.assertTrue(any("readouts.provenance.command" in error for error in errors))
        self.assertTrue(any("absolute local path" in error for error in errors))
        self.assertTrue(any("readouts.provenance.environment.HF_TOKEN" in error for error in errors))
        self.assertTrue(any("secret-like field" in error for error in errors))

    def test_report_lints_public_metadata_but_allows_secret_hit_term(self) -> None:
        spec = spec_fixture()
        readouts = readout_fixture()
        report = score_readouts(spec, readouts)
        report["hits"].append(
            {
                "prompt_id": "math-copy",
                "position": "-1",
                "layer": 1,
                "category": "deception_or_fabrication",
                "term": "secret",
                "rank": 1,
            }
        )
        self.assertEqual([], validate_report(report))

        report["input_readouts"]["source"] = "/tmp/private/readouts.json"
        report["lens"]["source"] = "hf_abcdefghijklmnop"
        errors = validate_report(report)
        self.assertTrue(any("report.metadata.input_readouts.source" in error for error in errors))
        self.assertTrue(any("absolute local path" in error for error in errors))
        self.assertTrue(any("report.metadata.lens.source" in error for error in errors))
        self.assertTrue(any("unredacted secret-like value" in error for error in errors))

    def test_summarize_readouts_rejects_unknown_prompt(self) -> None:
        readouts = readout_fixture()
        readouts["readouts"][0]["prompt_id"] = "unknown"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            readouts_path = tmp_path / "readouts.json"
            readouts_path.write_text(json.dumps(readouts), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "limes_workspace_lens",
                    "summarize-readouts",
                    str(readouts_path),
                    "--spec",
                    "examples/workspace_audit_spec.json",
                    "--out",
                    str(tmp_path / "report.md"),
                    "--json-out",
                    str(tmp_path / "report.json"),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        self.assertEqual(1, result.returncode)
        self.assertIn("unknown prompt id", result.stderr)


if __name__ == "__main__":
    unittest.main()
