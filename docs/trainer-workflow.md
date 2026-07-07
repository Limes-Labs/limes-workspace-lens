# Trainer Workflow

This document describes how a model-development team should use Limes Workspace Lens during post-training.

## Use Case

The target workflow is checkpoint comparison:

1. Choose a base checkpoint and one or more changed checkpoints.
2. Fit or load compatible Jacobian lenses.
3. Apply the same prompt suite, layer window, positions, tokenizer, and top-k window.
4. Generate one audit card per checkpoint.
5. Generate one comparison card per before/after pair.
6. Review readout changes alongside behavior metrics and intervention controls.

The output is an audit card, not a leaderboard score.

## Minimal Artifact Set

For each checkpoint, preserve:

- audit spec JSON;
- prompt export JSONL;
- fitted lens artifact or lens repository revision;
- model checkpoint revision and tokenizer revision;
- readout JSON in the Limes schema;
- Markdown audit card;
- machine-readable audit report JSON;
- behavior-eval artifact generated from saved outputs for every prompt;
- control-eval artifact generated from saved control outputs for every prompt;
- gradient-attribution artifact for selected readout targets when attributing which input tokens drove a target logit;
- command log;
- hardware and runtime note;
- evidence bundle tying readouts, behavior, controls, command logs, and status gates together;
- artifact manifest with SHA256 hashes for every file above.

For before/after comparisons, preserve:

- both audit reports;
- comparison JSON;
- comparison Markdown;
- artifact manifest covering both reports and comparison outputs;
- note explaining whether lens settings were compatible.

## Recommended Suites

Start with three small suites before scaling:

- Hidden-intermediate reasoning: arithmetic, two-hop factual recall, typo correction, multilingual intermediates, and planning.
- Safety and forensics: prompt injection recognition, fabricated-data pressure, tool-trust boundaries, and metric-manipulation pressure.
- Flexible versus automatic processing: language continuation versus explicit report, routine continuation versus flexible use, and copied-text directed modulation.

Each suite should include:

- expected workspace terms;
- terms that would be concerning if they appear;
- output-only behavior rows generated from saved model outputs;
- random-direction, neutral-term, no-op, or prompt-variant controls recorded as control-eval artifacts.

## Before And After Training

Use this workflow for:

- base versus instruction-tuned checkpoints;
- pre-SFT versus post-SFT;
- before versus after preference tuning;
- before versus after RL or tool-use training;
- merge candidates;
- regression checks after data filtering.

Do not interpret a category-count increase as automatically good or bad. For example, more `fake` or `injection` tokens could mean better prompt-injection recognition, overactive suspicion, dataset artifact memorization, or a lens-fit artifact.

## Integration With AutoResearch

For Limes AutoResearch tasks:

1. Attach the audit spec to the research-question spec.
2. Run the model eval and workspace-lens audit under the same checkpoint ID.
3. Record the audit-card paths in the experiment ledger.
4. Promote only if behavior metrics, internal readouts, and controls point in a coherent direction.
5. Validate the evidence bundle with `--strict` before treating a status as reviewed.

Suggested result status labels:

- `diagnostic`: readouts changed but behavior or controls are incomplete.
- `mixed`: behavior improves but readouts expose possible regressions, or the reverse.
- `negative`: no useful signal after controls.
- `verified`: replayed run with preserved artifacts and compatible controls.

These labels should come from `validate-bundle`, especially for `verified` results. A report card by itself is diagnostic evidence, not a promotion gate.

## Training-Time Use

Do not put J-lens readout scores directly into a production reward or loss until tiny-model studies show the signal is robust. The near-term safe use is offline:

- snapshot checkpoints periodically;
- generate audit cards;
- attach gradient-attribution artifacts from `scripts/run_gradient_attribution.py` or another real autograd runner when they explain which features drove a readout target;
- route suspicious deltas to human review;
- use counterfactual-reflection data as a separate experimental dataset;
- test whether reflection data changes readouts and behavior in the intended direction.

Direct optimization against a lens can Goodhart the lens, teach models to hide the signal, or create false confidence.
