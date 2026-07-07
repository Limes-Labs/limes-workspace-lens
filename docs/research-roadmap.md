# Research Roadmap

The goal is to make workspace-lens audits useful to model developers before attempting training-time optimization.

For the engineering completion plan, see `docs/system-completion-plan.md`. That document is the source of truth for missing production components, focused PR order, and acceptance gates.

## v0.1 - Audit Cards

Status: initial implementation.

- Dependency-free spec, readout, report, comparison, reflection-data, and intervention-plan schemas.
- CPU smoke path on synthetic fixtures.
- Optional `jlens` wrappers for real Hugging Face model environments.
- Documentation for non-claims and trainer workflow.

Acceptance gate:

- tests pass;
- smoke command runs from a fresh clone;
- README explains real-model path and evidence boundaries.

## v0.2 - Open-Weight Replication Pack

Add one small open-weight replication package:

- one model checkpoint;
- exact fitted-lens command;
- lens prompt corpus manifest;
- readout artifact;
- audit card;
- behavior checks;
- random-direction controls.

Candidate model families:

- Qwen small or mid-sized decoder;
- Gemma small decoder;
- Pythia or another small research-friendly checkpoint.

Acceptance gate:

- at least one hidden-intermediate or prompt-injection readout is reproduced on real model internals;
- fixture results are clearly separated from real results;
- negative and mixed examples are included.

## v0.3 - Checkpoint Regression Gates

Connect audit cards to Limes model workstreams:

- `limes-nanogpt` small training runs;
- AutoResearch ledgers;
- EuroBench prompt subsets;
- before/after LoRA or SFT runs.

Acceptance gate:

- before/after report comparisons are linked to behavior metrics;
- report deltas are not used as standalone quality scores.

## v0.4 - Intervention Controls

Add executable intervention support for models with hook access:

- coordinate swap;
- coordinate ablation;
- J-space versus non-J-space controls;
- answer-token versus intermediate-token swaps.

Acceptance gate:

- intervention effect exceeds random-direction controls on a small suite;
- failures are reported.

## v0.5 - Counterfactual Reflection Study

Test whether interrupted-reflection data changes internal readouts and behavior:

- generate reflection JSONL from locked specs;
- fine-tune a small model;
- compare before/after audit cards;
- run behavior checks;
- document Goodhart risks.

Acceptance gate:

- behavior and readout movement align on held-out prompts;
- no broad alignment claim is made.

## Later - Training-Time Monitoring

Only after the earlier gates:

- periodic J-lens snapshots during training;
- dashboard for drift and regression;
- data curation signals;
- experimental auxiliary losses in tiny settings.

No production reward use without strong evidence.
