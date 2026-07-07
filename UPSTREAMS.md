# Upstreams And Source Trail

This file records sources inspected for the initial public repo. It is an attribution and license boundary, not a claim that external authors endorse this work.

## Anthropic Global Workspace Paper

- Title: "Verbalizable Representations Form a Global Workspace in Language Models"
- URL: https://transformer-circuits.pub/2026/workspace/index.html
- Citation given by source: Gurnee et al., Transformer Circuits Thread, 2026
- Reuse scope: concepts, terminology, method framing, and public citation only.
- Source code copied: no.
- Notes: The paper defines the Jacobian lens, J-lens vectors, J-space, workspace-style functional properties, limitations, and counterfactual reflection training.

## Anthropic Research Summary

- URL: https://www.anthropic.com/research/global-workspace
- Inspection date: 2026-07-07
- Reuse scope: high-level summary and public source trail.
- Source code copied: no.
- Notes: Useful for public wording around J-space, non-claims about consciousness, and safety examples.

## anthropics/jacobian-lens

- URL: https://github.com/anthropics/jacobian-lens
- Inspected HEAD: `581d398613e5602a5af361e1c34d3a92ea82ba8e`
- License: Apache License 2.0, per repository metadata and README.
- Reuse scope: optional runtime dependency and compatibility target.
- Source code copied: no.
- Notes: This repository is not a fork. `scripts/fit_jlens.py` and `scripts/export_jlens_readouts.py` call the installed `jlens` package when the user installs it separately.

## External Commentary PDF

- Local path inspected: `/Users/francescogiannicola/Downloads/globalworkspace_externalcommentary_2July2026.pdf`
- Public URL discovered: https://www-cdn.anthropic.com/files/4zrzovbb/website/cc4be2488d65e54a6ed06492f8968398ddc18ebe.pdf
- Reuse scope: source-backed limitations, replication caution, and open questions.
- Source code copied: no.
- Notes: Commentary supports conservative interpretation: useful and important, but bounded, vocabulary-biased, and not a proof of phenomenal consciousness.

## Local Copies Supplied By User

- `/Users/francescogiannicola/Downloads/Verbalizable Representations Form a Global Workspace in Language Models.mhtml`
- `/Users/francescogiannicola/.codex/attachments/53989579-9fec-4051-8a31-a6af25e4874a/pasted-text.txt`
- Reuse scope: corroboration of public sources.
- Source code copied: no.

## Future Import Rule

Before importing code, datasets, prompt suites, screenshots, or generated HTML from any upstream, add:

- exact upstream URL and commit or version;
- license and copyright finding;
- file-level reuse list;
- reason this cannot remain an optional dependency;
- tests proving the imported asset is used as claimed.
