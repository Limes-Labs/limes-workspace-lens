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

## Minimum Real Claim

A minimum credible claim looks like:

> On model checkpoint X with lens artifact Y, prompt suite Z, and layer policy L, audit terms A appeared in the readouts at positions P. The same run produced behavior artifact B. Random-direction controls C did not produce the same effect. This supports a diagnostic hypothesis H, not a general model-quality claim.
