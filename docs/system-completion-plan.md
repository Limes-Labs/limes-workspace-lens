# System Completion Plan

This document defines the work needed for Limes Workspace Lens to become a serious tool for model developers and interpretability researchers.

## Goal

Build a reliable open-weight model audit workbench that can take real Jacobian-lens readouts from model checkpoints, preserve the evidence trail, produce reviewable audit cards, compare checkpoints safely, and support later controlled intervention and training-time studies.

The system is complete only when a user can:

1. Define a locked audit spec.
2. Export prompts for a real lens-fitting environment.
3. Fit or reference a lens with recorded provenance.
4. Export readouts in the Limes schema.
5. Validate all artifacts with checksums and compatibility rules.
6. Generate audit and checkpoint-comparison cards.
7. Pair internal readouts with behavior artifacts and controls.
8. Preserve negative, mixed, diagnostic, and verified results without ambiguity.

## Current Baseline

The v0.1 repository already has:

- dependency-free spec and readout validation;
- audit-card generation from readout JSON;
- checkpoint report comparison;
- counterfactual-reflection data generation;
- intervention-plan generation;
- optional wrappers for `anthropics/jacobian-lens`;
- synthetic fixtures used only for CI and onboarding;
- README, non-claims, trainer workflow, and roadmap docs;
- CI and smoke tests.

This is a useful scaffold, but it is not yet a complete production-quality workbench.

## Missing Production Components

### 1. Artifact Integrity Layer

Required:

- SHA256 manifests for specs, prompts, readouts, reports, comparisons, and behavior artifacts.
- Manifest validation that fails when an artifact is missing or has changed.
- Git commit, command, Python version, platform, and package metadata.
- Explicit `synthetic=false` gating for real claims.

Why it matters:

Audit cards are only useful if a reviewer can trace every claim back to immutable files.

### 2. Compatibility Gates For Comparisons

Required:

- Report schema validation.
- Comparison refusal by default when prompt IDs, top-k, lens source, layer policy, tokenizer policy, or audit categories are incompatible.
- `--allow-incompatible` escape hatch that records the incompatibility in the report.
- Tests for incompatible prompt suites and top-k windows.

Why it matters:

Before/after comparisons can become misleading if the two reports were generated under different lens or prompt policies.

### 3. Real-Model Adapter Hardening

Required:

- Optional dependency group documentation for real `jlens`, `torch`, and `transformers` use.
- Import-time failure messages that explain exactly what to install.
- Readout export tests using a fake in-process adapter, not a fake model claim.
- Device handling tests for CPU-only environments.
- Support for writing a model/lens manifest next to exported readouts.

Why it matters:

Researchers should not discover adapter fragility only after downloading model weights.

### 4. Behavior Artifact Pairing

Required:

- Schema for behavior outputs linked to prompt IDs.
- Audit-bundle validation requiring readouts plus behavior artifacts for non-diagnostic claims.
- Result status rules: `diagnostic`, `mixed`, `negative`, `verified`.
- Tests that reject `verified` status when controls or behavior artifacts are absent.

Why it matters:

Internal readouts are hypothesis-generation signals until tied to behavior and controls.

### 5. Control And Intervention Execution

Required:

- Executable intervention hooks for compatible model wrappers.
- Coordinate swap and ablation controls.
- Random-direction and neutral-token controls.
- Paired behavior summaries before and after intervention.
- Tests on small hookable toy modules before any model-scale claim.

Why it matters:

The strongest evidence in the source literature is causal. The repo should eventually support causal tests, not only readout inspection.

### 6. Visualization

Required:

- Static HTML or Markdown trace views for layer-by-position readouts.
- Pinned-token trajectories.
- Report links from audit card to raw rows.
- Accessibility and offline rendering.

Why it matters:

Human review needs more than aggregate counts. Researchers need to inspect where terms appear and disappear.

### 7. Open-Weight Replication Pack

Required:

- One small open-weight checkpoint with preserved model revision and license note.
- Lens-fit command and prompt corpus manifest.
- Real readout artifact with `synthetic=false`.
- Behavior artifacts for the same prompts.
- Negative or mixed cases included.
- Controls documented.

Why it matters:

The repo should not claim real utility until it demonstrates at least one real open-weight audit from model internals.

### 8. AutoResearch And Limes Integration

Required:

- AutoResearch result-card adapter.
- Ledger fields for workspace-lens artifact paths.
- EuroBench prompt subset integration plan.
- `limes-nanogpt` tiny-run integration plan.

Why it matters:

The workbench should become part of Limes model-development loops, not a standalone curiosity.

## Focused PR Plan

### PR 1: Production Completion Plan

Scope:

- Add this document.
- Link it from README and roadmap.
- Keep it documentation-only.

Acceptance:

- Link checks by inspection.
- Unit tests and smoke still pass.

### PR 2: Artifact Manifest And Checksum Validation

Scope:

- Add manifest builder and validator.
- Add CLI commands `build-manifest` and `validate-manifest`.
- Include tests for changed, missing, and valid files.
- Update docs.

Acceptance:

- Manifest validation catches tampered artifacts.
- Smoke command exercises manifest validation.

### PR 3: Report Compatibility Gates

Scope:

- Add report schema validation.
- Make comparisons fail on incompatible report settings by default.
- Add `--allow-incompatible`.
- Add tests for prompt, top-k, and audit-term incompatibility.

Acceptance:

- Incompatible comparisons fail loudly.
- Compatible comparisons preserve current behavior.

### PR 4: Behavior Artifact Pairing

Scope:

- Add behavior-artifact schema.
- Add audit-bundle validation.
- Add status-gate rules.
- Add docs for `diagnostic`, `mixed`, `negative`, and `verified`.

Acceptance:

- `verified` requires readouts, behavior artifact, controls, and manifest.
- Synthetic fixtures cannot be marked verified.

### PR 5: Real-Model Adapter Hardening

Scope:

- Improve optional `jlens` wrapper errors.
- Add model/lens manifest output.
- Add adapter tests with lightweight fakes.
- Document dependency groups and GPU/CPU boundaries.

Acceptance:

- CPU-only tests pass.
- Missing optional dependencies produce actionable errors.

### PR 6: Open-Weight Replication Pack

Scope:

- Run one real small open-weight audit.
- Commit readout and behavior artifacts if licenses and sizes allow.
- Include at least one negative or mixed result.

Acceptance:

- `synthetic=false` artifact is reproducible from documented commands.
- No model weights are committed.

## Verification Stack

Every PR should run:

```bash
python3 -m unittest discover -s tests
python3 -m py_compile limes_workspace_lens/*.py scripts/*.py
./scripts/run_smoke.sh
git diff --check
```

PRs that touch real-model wrappers should additionally run import-failure tests in an environment without optional ML dependencies.

PRs that add real artifacts should validate manifests and compare committed machine-readable artifacts against generated Markdown summaries.

## Explicit Non-Goals For Now

- No production reward model based on J-lens scores.
- No claim of consciousness, sentience, or hidden-goal detection.
- No cross-model leaderboard.
- No large model downloads in CI.
- No vendored upstream source without license review.

These can become future research tracks only after the artifact and validation layers are solid.
