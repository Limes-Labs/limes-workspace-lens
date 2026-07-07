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

Use `--expected-status verified` before promoting a result as verified. Verified bundles must include behavior and control artifacts, command logs, compute manifests, preserved hashes, compatible settings, and non-synthetic readouts.

## Minimum Real Claim

A minimum credible claim looks like:

> On model checkpoint X with lens artifact Y, prompt suite Z, and layer policy L, audit terms A appeared in the readouts at positions P. The same run produced behavior artifact B. Random-direction controls C did not produce the same effect. This supports a diagnostic hypothesis H, not a general model-quality claim.
