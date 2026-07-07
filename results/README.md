# Results

This directory is reserved for committed audit artifacts from real model runs.

The synthetic fixture in `examples/` is not a result. It exists only to test the parser, report generator, and smoke command.

For each real result, include:

- `*.json` readout artifact;
- `*.json` audit report;
- `*.md` audit card;
- command log;
- model and lens manifest;
- behavior-eval artifact when available;
- short interpretation note with limitations.

Use status labels:

- `diagnostic`
- `mixed`
- `negative`
- `verified`

Do not commit model weights or large lens artifacts unless the license and storage policy have been reviewed.
