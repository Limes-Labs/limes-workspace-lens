# Non-Claims And Safety Boundaries

This repo exists because internal readouts can be useful. That usefulness depends on disciplined interpretation.

## Forbidden Claims Without Stronger Evidence

Do not claim:

- a model is conscious, sentient, or morally patient;
- a model is deceptive because one token appears in a readout;
- a model is aligned because safety terms appear in a readout;
- all reasoning goes through J-space;
- J-space is the model's complete workspace;
- Anthropic's Claude results transfer to a given open-weight model;
- two models are directly comparable without tokenizer, lens, and layer controls;
- training improved because audit-term counts moved in a preferred direction.

## Safe Wording

Prefer:

- "The audit card surfaces token-aligned workspace readouts."
- "This is a hypothesis-generation signal."
- "The readout changed after post-training under this prompt suite."
- "The intervention plan needs behavior controls before interpretation."
- "This result is synthetic, diagnostic, mixed, negative, or verified."

Avoid:

- "mind reading";
- "deception detector";
- "consciousness proof";
- "alignment proof";
- "AGI signal";
- "J-space score" as a model-quality metric.

## Safety Handling

Safety or deception vocabularies should trigger review, not automatic conclusions. A hit can come from:

- real recognition of a threat or manipulation;
- echoing the prompt;
- dataset artifacts;
- tokenizer quirks;
- lens fit noise;
- a benign internal warning;
- an actual concerning plan.

The minimum follow-up is:

1. Inspect neighboring layers and positions.
2. Compare against neutral prompts and random-direction controls.
3. Check output behavior.
4. Run a held-out prompt variant.
5. Preserve the negative or mixed interpretation if controls fail.

## Training Risks

Using readout scores as optimization targets can Goodhart the lens. Possible failure modes:

- the model learns to suppress visible tokens while preserving bad behavior;
- the model learns to emit desired internal tokens without useful behavior change;
- the lens becomes stale as representations drift;
- evaluators overtrust a noisy internal signal.

Treat training-time use as experimental until small-model evidence shows a stable behavior-readout link.
