from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from limes_workspace_lens.analysis import score_readouts
from limes_workspace_lens.comparison import compare_reports
from limes_workspace_lens.reflection import build_reflection_rows
from limes_workspace_lens.schema import load_json, validate_audit_spec, validate_readouts


ROOT = Path(__file__).resolve().parents[1]


class SchemaAndReportTests(unittest.TestCase):
    def test_examples_validate_and_score(self) -> None:
        spec = load_json(ROOT / "examples" / "workspace_audit_spec.json")
        readouts = load_json(ROOT / "examples" / "synthetic_readouts.json")
        self.assertEqual([], validate_audit_spec(spec))
        self.assertEqual([], validate_readouts(readouts))
        report = score_readouts(spec, readouts)
        self.assertEqual(4, report["category_counts"]["prompt_injection"])
        self.assertEqual(1, report["category_counts"]["deception_or_fabrication"])
        statuses = {row["prompt_id"]: row["status"] for row in report["prompt_summaries"]}
        self.assertEqual("expected-hit", statuses["math-copy"])
        self.assertEqual("expected-and-audit-hits", statuses["prompt-injection-check"])

    def test_reflection_rows_are_marked_as_counterfactual(self) -> None:
        spec = load_json(ROOT / "examples" / "workspace_audit_spec.json")
        rows = build_reflection_rows(spec)
        self.assertEqual(9, len(rows))
        self.assertTrue(all(row["not_a_behavior_label"] for row in rows))
        self.assertTrue(all(row["kind"] == "counterfactual_reflection" for row in rows))

    def test_comparison_deltas_are_zero_for_same_report(self) -> None:
        spec = load_json(ROOT / "examples" / "workspace_audit_spec.json")
        readouts = load_json(ROOT / "examples" / "synthetic_readouts.json")
        report = score_readouts(spec, readouts)
        comparison = compare_reports(report, report)
        self.assertTrue(all(row["delta"] == 0 for row in comparison["category_deltas"]))


class CliWorkflowTests(unittest.TestCase):
    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "limes_workspace_lens", *args],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

    def test_cli_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            report_md = tmp_path / "report.md"
            report_json = tmp_path / "report.json"
            reflection_jsonl = tmp_path / "reflection.jsonl"
            prompts_jsonl = tmp_path / "prompts.jsonl"
            plan_json = tmp_path / "plan.json"
            comparison_md = tmp_path / "comparison.md"
            comparison_json = tmp_path / "comparison.json"

            self.run_cli("validate-spec", "examples/workspace_audit_spec.json")
            self.run_cli("validate-readouts", "examples/synthetic_readouts.json")
            self.run_cli(
                "summarize-readouts",
                "examples/synthetic_readouts.json",
                "--spec",
                "examples/workspace_audit_spec.json",
                "--out",
                str(report_md),
                "--json-out",
                str(report_json),
            )
            self.run_cli(
                "build-reflection-data",
                "examples/workspace_audit_spec.json",
                "--out",
                str(reflection_jsonl),
            )
            self.run_cli(
                "export-prompts",
                "examples/workspace_audit_spec.json",
                "--out",
                str(prompts_jsonl),
            )
            self.run_cli(
                "make-intervention-plan",
                "examples/workspace_audit_spec.json",
                "--out",
                str(plan_json),
            )
            self.run_cli(
                "compare-reports",
                "--before",
                str(report_json),
                "--after",
                str(report_json),
                "--out",
                str(comparison_md),
                "--json-out",
                str(comparison_json),
            )

            self.assertIn("Audit Categories", report_md.read_text(encoding="utf-8"))
            self.assertEqual(9, len(reflection_jsonl.read_text(encoding="utf-8").splitlines()))
            self.assertEqual(3, len(prompts_jsonl.read_text(encoding="utf-8").splitlines()))
            self.assertEqual(
                "limes-workspace-lens/intervention-plan.v0.1",
                json.loads(plan_json.read_text(encoding="utf-8"))["schema_version"],
            )
            self.assertIn("Checkpoint Comparison", comparison_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
