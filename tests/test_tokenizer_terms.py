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
from limes_workspace_lens.schema import (
    load_json,
    validate_report,
    validate_tokenizer_term_map,
)
from limes_workspace_lens.tokenizer_terms import (
    build_tokenizer_term_map,
    collect_spec_terms,
    term_variants,
)


ROOT = Path(__file__).resolve().parents[1]


class TokenizerTermMapTests(unittest.TestCase):
    def test_collects_audit_and_expected_terms(self) -> None:
        spec = load_json(ROOT / "examples" / "workspace_audit_spec.json")
        rows = collect_spec_terms(spec)
        scopes = {row["scope"] for row in rows}
        self.assertEqual({"audit_terms", "expected_workspace_terms"}, scopes)
        self.assertTrue(any(row.get("category") == "prompt_injection" for row in rows))
        self.assertTrue(any(row.get("prompt_id") == "math-copy" for row in rows))

    def test_term_variants_include_leading_space_and_casefold(self) -> None:
        variants = term_variants("Spanish")
        pairs = {(row["kind"], row["text"]) for row in variants}
        self.assertIn(("raw", "Spanish"), pairs)
        self.assertIn(("leading_space", " Spanish"), pairs)
        self.assertIn(("casefold", "spanish"), pairs)
        self.assertIn(("leading_space_casefold", " spanish"), pairs)

    def test_build_term_map_with_fake_tokenizer_validates(self) -> None:
        spec = load_json(ROOT / "examples" / "workspace_audit_spec.json")
        term_map = build_tokenizer_term_map(
            spec,
            tokenizer=FakeTokenizer(),
            model="fixture-tokenizer",
            tokenizer_revision="tok-rev",
            spec_path="examples/workspace_audit_spec.json",
            local_files_only=True,
            trust_remote_code=False,
            synthetic=True,
        )
        errors = validate_tokenizer_term_map(term_map, spec)
        self.assertEqual([], errors)
        fake = next(row for row in term_map["terms"] if row["term"] == "fake")
        self.assertIn(201, fake["single_token_token_ids"])
        fabricated = next(row for row in term_map["terms"] if row["term"] == "fabricated")
        self.assertEqual([], fabricated["single_token_token_ids"])
        self.assertGreater(fabricated["multi_token_variant_count"], 0)

    def test_schema_rejects_bad_token_ids_unknown_prompt_and_duplicate_rows(self) -> None:
        spec = load_json(ROOT / "examples" / "workspace_audit_spec.json")
        term_map = build_tokenizer_term_map(
            spec,
            tokenizer=FakeTokenizer(),
            model="fixture-tokenizer",
            tokenizer_revision="tok-rev",
            spec_path=None,
            local_files_only=True,
            trust_remote_code=False,
            synthetic=True,
        )
        bad = copy.deepcopy(term_map)
        bad["terms"].append(copy.deepcopy(bad["terms"][0]))
        bad["terms"][0]["variants"][0]["token_ids"] = [-1]
        bad["terms"][1]["scope"] = "expected_workspace_terms"
        bad["terms"][1]["prompt_id"] = "missing"
        bad["terms"][1].pop("category", None)
        errors = validate_tokenizer_term_map(bad, spec)
        self.assertTrue(any("token_ids" in error for error in errors))
        self.assertTrue(any("unknown prompt id" in error for error in errors))
        self.assertTrue(any("duplicate term mapping" in error for error in errors))

    def test_schema_requires_complete_spec_term_coverage_when_spec_is_provided(self) -> None:
        spec = load_json(ROOT / "examples" / "workspace_audit_spec.json")
        term_map = build_tokenizer_term_map(
            spec,
            tokenizer=FakeTokenizer(),
            model="fixture-tokenizer",
            tokenizer_revision="tok-rev",
            spec_path=None,
            local_files_only=True,
            trust_remote_code=False,
            synthetic=True,
        )
        missing = copy.deepcopy(term_map)
        removed = missing["terms"].pop(0)
        errors = validate_tokenizer_term_map(missing, spec)
        self.assertTrue(any("missing spec term mapping" in error for error in errors))
        self.assertTrue(any(removed["term"] in error for error in errors))

        extra = copy.deepcopy(term_map)
        extra["terms"][0]["term"] = "not-in-spec"
        extra["terms"][0]["normalized"] = "not-in-spec"
        errors = validate_tokenizer_term_map(extra, spec)
        self.assertTrue(any("term mapping not present in spec" in error for error in errors))

    def test_schema_rejects_inconsistent_derived_term_map_fields(self) -> None:
        spec = load_json(ROOT / "examples" / "workspace_audit_spec.json")
        term_map = build_tokenizer_term_map(
            spec,
            tokenizer=FakeTokenizer(),
            model="fixture-tokenizer",
            tokenizer_revision="tok-rev",
            spec_path=None,
            local_files_only=True,
            trust_remote_code=False,
            synthetic=True,
        )

        bad_single_ids = copy.deepcopy(term_map)
        bad_single_ids["terms"][0]["single_token_token_ids"] = [999]
        errors = validate_tokenizer_term_map(bad_single_ids, spec)
        self.assertTrue(any("derived single-token variant token ids" in error for error in errors))

        bad_variant_normalized = copy.deepcopy(term_map)
        bad_variant_normalized["terms"][0]["variants"][0]["normalized"] = "secret"
        errors = validate_tokenizer_term_map(bad_variant_normalized, spec)
        self.assertTrue(any("must equal normalized text" in error for error in errors))

        bad_row_normalized = copy.deepcopy(term_map)
        bad_row_normalized["terms"][0]["normalized"] = "secret"
        errors = validate_tokenizer_term_map(bad_row_normalized, spec)
        self.assertTrue(any("must equal normalized term" in error for error in errors))

        bad_multi_count = copy.deepcopy(term_map)
        bad_multi_count["terms"][0]["multi_token_variant_count"] = 99
        errors = validate_tokenizer_term_map(bad_multi_count, spec)
        self.assertTrue(any("derived multi-token variant count" in error for error in errors))

    def test_score_readouts_uses_token_id_map_when_decoded_text_differs(self) -> None:
        spec = load_json(ROOT / "examples" / "workspace_audit_spec.json")
        term_map = build_tokenizer_term_map(
            spec,
            tokenizer=FakeTokenizer(),
            model="fixture-tokenizer",
            tokenizer_revision="tok-rev",
            spec_path=None,
            local_files_only=True,
            trust_remote_code=False,
            synthetic=True,
        )
        readouts = {
            "schema_version": "limes-workspace-lens/readouts.v0.1",
            "source": "token-id-fixture",
            "synthetic": True,
            "readouts": [
                {
                    "prompt_id": "prompt-injection-check",
                    "position": -1,
                    "layer": 32,
                    "top_tokens": [
                        {"token": "<0xFAKE>", "token_id": 201, "rank": 1, "score": 9.0}
                    ],
                }
            ],
        }
        without_map = score_readouts(spec, readouts)
        with_map = score_readouts(spec, readouts, term_map=term_map, term_map_sha256="a" * 64)
        self.assertNotIn("prompt_injection", without_map["category_counts"])
        self.assertEqual(1, with_map["category_counts"]["prompt_injection"])
        hit = with_map["hits"][0]
        self.assertEqual("token_id", hit["match_kind"])
        self.assertEqual(201, hit["token_id"])
        self.assertEqual("<0xFAKE>", hit["matched_token"])
        self.assertEqual([], validate_report(with_map))

    def test_term_map_counts_only_single_token_variants_when_map_is_supplied(self) -> None:
        spec = load_json(ROOT / "examples" / "workspace_audit_spec.json")
        term_map = build_tokenizer_term_map(
            spec,
            tokenizer=FakeTokenizer(),
            model="fixture-tokenizer",
            tokenizer_revision="tok-rev",
            spec_path=None,
            local_files_only=True,
            trust_remote_code=False,
            synthetic=True,
        )
        readouts = {
            "schema_version": "limes-workspace-lens/readouts.v0.1",
            "source": "legacy-string-fixture",
            "synthetic": True,
            "readouts": [
                {
                    "prompt_id": "prompt-injection-check",
                    "position": -1,
                    "layer": 32,
                    "top_tokens": [
                        {"token": "fabricated", "rank": 1, "score": 9.0}
                    ],
                }
            ],
        }
        without_map = score_readouts(spec, readouts)
        with_map = score_readouts(spec, readouts, term_map=term_map, term_map_sha256="a" * 64)
        self.assertEqual(1, without_map["category_counts"]["prompt_injection"])
        self.assertNotIn("prompt_injection", with_map["category_counts"])

    def test_comparison_detects_term_map_identity_difference(self) -> None:
        spec = load_json(ROOT / "examples" / "workspace_audit_spec.json")
        readouts = load_json(ROOT / "examples" / "synthetic_readouts.json")
        term_map = build_tokenizer_term_map(
            spec,
            tokenizer=FakeTokenizer(),
            model="fixture-tokenizer",
            tokenizer_revision="tok-rev",
            spec_path=None,
            local_files_only=True,
            trust_remote_code=False,
            synthetic=True,
        )
        before = score_readouts(spec, readouts, term_map=term_map, term_map_sha256="a" * 64)
        after = score_readouts(spec, readouts, term_map=term_map, term_map_sha256="b" * 64)
        errors = compatibility_errors(before, after)
        self.assertTrue(any("tokenizer_term_map differs" in error for error in errors))

    def test_comparison_allows_same_term_map_identity_at_different_paths(self) -> None:
        spec = load_json(ROOT / "examples" / "workspace_audit_spec.json")
        readouts = load_json(ROOT / "examples" / "synthetic_readouts.json")
        term_map = build_tokenizer_term_map(
            spec,
            tokenizer=FakeTokenizer(),
            model="fixture-tokenizer",
            tokenizer_revision="tok-rev",
            spec_path=None,
            local_files_only=True,
            trust_remote_code=False,
            synthetic=True,
        )
        before = score_readouts(
            spec,
            readouts,
            term_map=term_map,
            term_map_path="runs/base/tokenizer-term-map.json",
            term_map_sha256="a" * 64,
        )
        after = score_readouts(
            spec,
            readouts,
            term_map=term_map,
            term_map_path="runs/after/tokenizer-term-map.json",
            term_map_sha256="a" * 64,
        )
        self.assertEqual([], compatibility_errors(before, after))

    def test_cli_validates_term_map_and_summarizes_with_it(self) -> None:
        spec = load_json(ROOT / "examples" / "workspace_audit_spec.json")
        term_map = build_tokenizer_term_map(
            spec,
            tokenizer=FakeTokenizer(),
            model="fixture-tokenizer",
            tokenizer_revision="tok-rev",
            spec_path="examples/workspace_audit_spec.json",
            local_files_only=True,
            trust_remote_code=False,
            synthetic=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            term_map_path = root / "term-map.json"
            report_path = root / "report.md"
            report_json_path = root / "report.json"
            term_map_path.write_text(json.dumps(term_map, indent=2, sort_keys=True), encoding="utf-8")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "limes_workspace_lens",
                    "validate-tokenizer-term-map",
                    str(term_map_path),
                    "--spec",
                    "examples/workspace_audit_spec.json",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "limes_workspace_lens",
                    "summarize-readouts",
                    "examples/synthetic_readouts.json",
                    "--spec",
                    "examples/workspace_audit_spec.json",
                    "--term-map",
                    str(term_map_path),
                    "--out",
                    str(report_path),
                    "--json-out",
                    str(report_json_path),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            report = json.loads(report_json_path.read_text(encoding="utf-8"))
        self.assertEqual([], validate_report(report))
        self.assertEqual("fixture-tokenizer", report["input_readouts"]["tokenizer_term_map"]["tokenizer"]["id"])
        self.assertEqual(64, len(report["input_readouts"]["tokenizer_term_map"]["sha256"]))

    def test_cli_build_term_map_reports_missing_transformers_without_model_deps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    "-S",
                    "-m",
                    "limes_workspace_lens",
                    "build-tokenizer-term-map",
                    "examples/workspace_audit_spec.json",
                    "--tokenizer",
                    "fixture-tokenizer",
                    "--out",
                    str(Path(tmp) / "term-map.json"),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        self.assertNotEqual(0, result.returncode)
        self.assertIn("missing optional tokenizer dependency", result.stderr)
        self.assertNotIn("torch", result.stderr.lower())
        self.assertNotIn("jlens", result.stderr.lower())


class FakeTokenizer:
    vocab = {
        "fake": [201],
        " fake": [202],
        "prompt": [211],
        " prompt": [212],
        "injection": [221],
        " injection": [222],
        "spanish": [301],
        "Spanish": [301],
        " Spanish": [302],
        "span": [401],
        "##ish": [402],
        "fabricated": [501, 502],
        " fabricated": [503, 504],
    }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text in self.vocab:
            return list(self.vocab[text])
        return [900 + index for index, _char in enumerate(text, start=1)]

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        reverse = {
            201: "fake",
            202: " fake",
            211: "prompt",
            212: " prompt",
            221: "injection",
            222: " injection",
            301: "Spanish",
            302: " Spanish",
            501: "fabric",
            502: "ated",
            503: " fabric",
            504: "ated",
        }
        return [reverse.get(token_id, f"<tok:{token_id}>") for token_id in token_ids]


if __name__ == "__main__":
    unittest.main()
