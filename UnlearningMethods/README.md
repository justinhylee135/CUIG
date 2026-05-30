# Unlearning Methods

This directory contains the trainable unlearning methods used by CUIG. Method code lives here, while reusable regularizers live under [`../Regularizers`](../Regularizers/README.md), benchmark evaluation lives under [`../Evaluation`](../Evaluation/README.md), and full experiment launchers live under [`../BashScripts`](../BashScripts/README.md).

At the moment, the implemented method is [`ConAbl`](ConAbl/README.md).

## Layout

```text
UnlearningMethods/
|-- ConAbl/
`-- README.md
```

## ConAbl

[`ConAbl`](ConAbl/README.md) is the current unlearning method used across independent, sequential, and simultaneous CUIG experiments. It trains selected model parameters so target concepts are mapped toward anchor concepts, with optional anchor preservation and regularization.

Typical use cases:

- independent unlearning: train one checkpoint per target concept
- sequential unlearning: continue from the previous unlearned checkpoint with `--unet_ckpt`
- simultaneous unlearning: train on a multi-concept configuration and optionally evaluate during training
- regularized unlearning: combine ConAbl with Weight, Projection, SelFT, or hybrid regularizers

See [`ConAbl/README.md`](ConAbl/README.md) for the method layout, direct training commands, supported concept inputs, anchor dataset behavior, trainable parameter groups, regularizer flags, and output conventions.

## Adding a New Method

Add new methods as sibling directories under `UnlearningMethods/`.

Recommended structure:

- provide a method-specific training entrypoint
- keep argument parsing, data setup, model setup, and training utilities split under `src/`
- use regularizers from `Regularizers/` instead of duplicating shared logic
- write checkpoints under the user-provided `--output_dir`
- add BashScripts launchers for full experiment pipelines
- add a short method README if the method requires custom data or setup
