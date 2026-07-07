# Security Policy

Limes Workspace Lens is an artifact-validation and review tool for model-internal audit evidence. It should not preserve private tokens, local filesystem details, private model paths, or raw credentials in public artifacts.

## Reporting

Please report suspected vulnerabilities or accidental data-exposure risks by opening a private security advisory on GitHub, or by contacting the Limes Labs maintainers through the organization contact listed on GitHub.

Do not open a public issue containing secrets, private model paths, private dataset paths, command logs with credentials, or exploitable details.

## Artifact Hygiene

Before publishing evidence bundles:

- run `python3 -m limes_workspace_lens validate-bundle BUNDLE --root ROOT --strict`;
- validate command logs, compute manifests, and lens identity artifacts directly when iterating on them;
- redact tokens and credentials as `<redacted>` or environment placeholders such as `<env:HF_TOKEN>`;
- keep artifact paths bundle-relative;
- avoid absolute local paths such as `/Users/...`, `/home/...`, `/tmp/...`, `/scratch/...`, `/private/...`, `file://...`, or Windows drive paths;
- preserve SHA256 hashes for artifacts required by a `verified` bundle.

The validators reject common secret-like values and local path leaks in command logs, compute manifests, lens identity artifacts, artifact manifests, readout metadata/provenance, report identity metadata, behavior/control generation metadata, and strict bundle-loaded artifacts. They intentionally avoid linting token readout rows, audit terms, and prompt text because those fields can legitimately contain security-like concepts under study.

These checks are a guardrail, not a replacement for a private release review. Treat prompt corpora, raw model outputs, local checkpoint paths, `trust_remote_code` runs, and model downloads as trust-boundary events before publishing.
