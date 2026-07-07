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

The v0.1 repository now has:

- dependency-free spec, readout, report, manifest, comparison, evidence-bundle, reflection-data, and intervention-plan validation;
- audit-card generation from readout JSON;
- checkpoint report comparison with compatibility gates;
- artifact manifests with SHA256 validation;
- evidence-bundle validation for `diagnostic`, `mixed`, `negative`, and `verified` status rules;
- counterfactual-reflection data generation;
- intervention-plan generation;
- hardened optional wrappers for `anthropics/jacobian-lens`;
- synthetic fixtures used only for CI and onboarding;
- README, non-claims, trainer workflow, and roadmap docs;
- CI and integrated smoke tests.

This is now a useful artifact-contract and audit-card workbench, but it is not yet a complete production-quality Jacobian-lens audit tool because it still lacks runnable behavior/control generation, real open-weight evidence, and executable interventions.

## Live Todo

### Done

- [x] Public completion plan and workflow docs.
- [x] Artifact manifest builder and validator.
- [x] Comparison compatibility gates and `--allow-incompatible` diagnostics.
- [x] Hardened spec/readout/report validation.
- [x] Evidence-bundle validator with status gates and strict artifact checks.
- [x] Real-model adapter hardening for optional dependency errors, pinned revisions, local replay flags, device preflight, and provenance.
- [x] Integrated smoke path covering manifests, comparison, evidence bundle validation, and adapter help commands.

### Remaining

- [ ] Runnable `behavior-eval.v0.1` and `control-eval.v0.1` artifact generators.
- [ ] One real open-weight replication pack with `synthetic=false` readouts, behavior artifacts, controls, manifests, command logs, compute manifest, and evidence bundle.
- [ ] Executable intervention runtime for hookable toy modules before model-scale intervention claims.
- [ ] Tokenizer-aware term mapping and lens-fit diagnostics.
- [ ] Manual/canary real-model adapter workflow against a pinned tiny public checkpoint.
- [ ] Package build, wheel-install, and console-script CI.
- [ ] Security/reproducibility hardening for secret/path linting, symlink escape checks, and command-log redaction.
- [ ] Limes integration adapters for AutoResearch, EuroBench subsets, and `limes-nanogpt` checkpoints.
- [ ] Static offline review visualization and schema registry/golden fixtures.

## Missing Production Components

### 1. Runnable Behavior And Control Artifacts

Required:

- `run-behavior-eval` command that turns a locked audit spec into a behavior artifact.
- `run-control-eval` command for random-direction, neutral-token, no-op, or prompt-variant controls.
- Prompt coverage checks for every prompt in the claimed bundle.
- Generation config, seed, model identity, metric definitions, and command log fields.

Why it matters:

Evidence bundles currently validate behavior/control artifacts when present, but the repo does not yet help users create those artifacts.

### 2. Open-Weight Replication Pack

Required:

- One small open-weight checkpoint with preserved model revision and license note.
- Lens-fit command and prompt corpus manifest.
- Real readout artifact with `synthetic=false`.
- Behavior artifacts for the same prompts.
- Negative or mixed cases included.
- Controls documented.

Why it matters:

The repo should not claim real utility until it demonstrates at least one real open-weight audit from model internals.

### 3. Control And Intervention Execution

Required:

- Executable intervention hooks for compatible model wrappers.
- Coordinate swap and ablation controls.
- Random-direction and neutral-token controls.
- Paired behavior summaries before and after intervention.
- Tests on small hookable toy modules before any model-scale claim.

Why it matters:

The strongest evidence in the source literature is causal. The repo should eventually support causal tests, not only readout inspection.

### 4. Tokenizer-Aware Term Mapping And Lens-Fit Diagnostics

Required:

- Tokenizer-aware aliases for leading-space, BPE, split-token, casing, and Unicode variants.
- Multi-token concept matching for audit vocabularies.
- Lens-fit diagnostic artifact recording fit corpus, held-out checks, seed, layer policy, and ablations.
- Validation that model-family claims require fit-quality evidence.

Why it matters:

Exact decoded-token string matching is useful for fixtures, but too brittle for serious tokenizer-dependent claims.

### 5. Packaging, Security, And Reproducibility Hardening

Required:

- Build and install wheel in CI.
- Run console-script smoke outside the repo root.
- Add Python version matrix.
- Add `SECURITY.md`, artifact secret/path linter, command-log redaction rules, and symlink escape tests.
- Keep `trust_remote_code` and model-download risks explicit.

Why it matters:

Model-development teams need the tool to install cleanly, avoid accidental data leaks, and preserve replayable artifacts.

### 6. Visualization

Required:

- Static HTML or Markdown trace views for layer-by-position readouts.
- Pinned-token trajectories.
- Report links from audit card to raw rows.
- Accessibility and offline rendering.

Why it matters:

Human review needs more than aggregate counts. Researchers need to inspect where terms appear and disappear.

### 7. AutoResearch And Limes Integration

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

### PR 4: Schema And Readout Hardening

Scope:

- Tighten spec, readout, and report validation.
- Reject unknown prompt IDs during summarization.
- Add negative schema tests.

Acceptance:

- Malformed artifacts fail loudly.
- Existing examples remain valid.

### PR 5: Evidence Bundle Validation

Scope:

- Add audit-bundle validation.
- Add status-gate rules.
- Add docs for `diagnostic`, `mixed`, `negative`, and `verified`.

Acceptance:

- `verified` requires readouts, behavior artifact, controls, and manifest.
- Synthetic fixtures cannot be marked verified.

### PR 6: Real-Model Adapter Hardening

Scope:

- Improve optional `jlens` wrapper errors.
- Add model/lens manifest output.
- Add adapter tests with lightweight fakes.
- Document dependency groups and GPU/CPU boundaries.

Acceptance:

- CPU-only tests pass.
- Missing optional dependencies produce actionable errors.

### PR 7: Integrated Landing Gate

Scope:

- Extend smoke coverage across merged command surfaces.
- Validate a diagnostic evidence bundle in the smoke run.
- Update this todo after merged foundations.

Acceptance:

- Smoke exercises manifest validation, evidence-bundle validation, comparison gates, and adapter help commands.

### PR 8: Behavior And Control Artifact Generators

Scope:

- Add behavior/control artifact schemas and commands.
- Keep default implementation deterministic and dependency-light.
- Pair generated artifacts with evidence bundles.

Acceptance:

- Every prompt in a spec receives behavior/control rows.
- Verified bundles fail when rows are missing.

### PR 9: Open-Weight Replication Pack

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
