# Evaluation Contract

Workspace-lens evaluation must pair internal readouts with external behavior.

## Required Controls

- Random-direction or neutral-token controls for interventions.
- Output-only behavior checks for every internal-readout claim.
- Prompt variants to reduce single-prompt artifacts.
- Tokenizer-aware term mapping for multi-token concepts.
- Fit-quality ablations when making model-family claims.

## Report Status

- `diagnostic`: useful signal, incomplete controls.
- `mixed`: evidence points in more than one direction.
- `negative`: tested signal did not survive controls.
- `verified`: replayed, preserved, behavior-linked, and control-backed.

Status labels should be backed by an evidence bundle, not assigned by prose alone:

```bash
python3 -m limes_workspace_lens validate-bundle results/run/evidence-bundle.json --root results/run --strict
```

Use `--expected-status verified` before promoting a result as verified. `mixed`, `negative`, and `verified` bundles require `--strict`; verified bundles must include passing behavior/control rows, command logs, compute manifests, preserved hashes, compatible settings, and non-synthetic readouts.

Behavior and control artifacts can be generated from saved output JSONL without loading a model backend:

```bash
python3 -m limes_workspace_lens run-behavior-eval SPEC --responses outputs.jsonl --out behavior.json \
  --tokenizer-revision TOKENIZER_REV --lens-revision LENS_REV --fit-procedure FIT_LABEL --position-policy POSITION_POLICY
python3 -m limes_workspace_lens run-control-eval SPEC --responses controls.jsonl --control-kind prompt_variant --out control.json \
  --tokenizer-revision TOKENIZER_REV --lens-revision LENS_REV --fit-procedure FIT_LABEL --position-policy POSITION_POLICY
```

These commands score preserved outputs. They are not model runners and do not prove that an intervention or random-direction control was executed.

## Minimum Real Claim

A minimum credible claim looks like:

> On model checkpoint X with lens artifact Y, prompt suite Z, and layer policy L, audit terms A appeared in the readouts at positions P. The same run produced behavior artifact B. Random-direction controls C did not produce the same effect. This supports a diagnostic hypothesis H, not a general model-quality claim.
