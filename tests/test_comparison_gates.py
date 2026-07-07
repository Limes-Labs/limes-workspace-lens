from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from limes_workspace_lens.analysis import score_readouts
from limes_workspace_lens.comparison import compatibility_errors
from limes_workspace_lens.schema import load_json, validate_report


ROOT = Path(__file__).resolve().parents[1]


def fixture_report() -> dict:
    spec = load_json(ROOT / "examples" / "workspace_audit_spec.json")
    readouts = load_json(ROOT / "examples" / "synthetic_readouts.json")
    return score_readouts(spec, readouts)


class ComparisonGateTests(unittest.TestCase):
    def run_cli(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "limes_workspace_lens", *args],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=check,
        )

    def write_report(self, path: Path, report: dict) -> None:
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def test_report_schema_validation_rejects_bad_top_k(self) -> None:
        report = fixture_report()
        report["top_k"] = 0
        errors = validate_report(report)
        self.assertTrue(any("top_k" in error for error in errors))

    def test_compatibility_detects_top_k_prompt_and_lens_differences(self) -> None:
        before = fixture_report()
        after = copy.deepcopy(before)
        after["top_k"] = 5
        after["lens"]["source"] = "other-lens"
        after["prompt_summaries"] = after["prompt_summaries"][:-1]
        errors = compatibility_errors(before, after)
        self.assertTrue(any("top_k differs" in error for error in errors))
        self.assertTrue(any("lens.source differs" in error for error in errors))
        self.assertTrue(any("prompt suite differs" in error for error in errors))

    def test_cli_fails_closed_for_incompatible_reports(self) -> None:
        before = fixture_report()
        after = copy.deepcopy(before)
        after["top_k"] = 5
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before_path = root / "before.json"
            after_path = root / "after.json"
            out_path = root / "comparison.md"
            json_path = root / "comparison.json"
            self.write_report(before_path, before)
            self.write_report(after_path, after)
            failed = self.run_cli(
                "compare-reports",
                "--before",
                str(before_path),
                "--after",
                str(after_path),
                "--out",
                str(out_path),
                "--json-out",
                str(json_path),
                check=False,
            )
            self.assertNotEqual(0, failed.returncode)
            self.assertIn("top_k differs", failed.stderr)

            self.run_cli(
                "compare-reports",
                "--before",
                str(before_path),
                "--after",
                str(after_path),
                "--out",
                str(out_path),
                "--json-out",
                str(json_path),
                "--allow-incompatible",
            )
            comparison = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual("incompatible-allowed", comparison["compatibility"]["status"])

    def test_cli_rejects_non_positive_top_k(self) -> None:
        failed = self.run_cli(
            "summarize-readouts",
            "examples/synthetic_readouts.json",
            "--spec",
            "examples/workspace_audit_spec.json",
            "--out",
            "/tmp/unused.md",
            "--json-out",
            "/tmp/unused.json",
            "--top-k",
            "0",
            check=False,
        )
        self.assertNotEqual(0, failed.returncode)
        self.assertIn("positive integer", failed.stderr)


if __name__ == "__main__":
    unittest.main()
