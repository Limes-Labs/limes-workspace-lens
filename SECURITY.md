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
- avoid absolute local paths such as `/Users/...`, `/home/...`, `/private/...`, `file://...`, or Windows drive paths;
- preserve SHA256 hashes for artifacts required by a `verified` bundle.

The validators reject common secret-like values and local path leaks in command logs, compute manifests, and lens identity artifacts. Broader public-artifact linting for manifests, readout provenance, and behavior/control metadata is tracked in the completion plan.
