# AutoResearch Integration

Limes Workspace Lens can be attached to Limes AutoResearch tasks as a diagnostic sidecar.

Recommended task fields:

- audit spec path;
- prompt JSONL path;
- model checkpoint;
- lens artifact or lens source;
- readout artifact;
- audit report JSON;
- audit card Markdown;
- comparison card when a prior checkpoint exists.

Promotion rule:

Do not promote a model change because the workspace-lens report looks better. Promote only when behavior metrics, internal readouts, controls, and review notes agree.
