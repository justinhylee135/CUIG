# Unlearning Methods

This directory contains the trainable unlearning methods used by CUIG. Method code lives here, while reusable regularizers live under [`../Regularizers`](../Regularizers/README.md), benchmark evaluation lives under [`../Evaluation`](../Evaluation/README.md), and full experiment launchers live under [`../BashScripts`](../BashScripts/README.md).

Implemented methods include [`ConAbl`](ConAbl/README.md), [`SculpMem`](SculpMem/README.md), and [`ESD-SDXL`](ESD-SDXL/README.md).

## Layout

```text
UnlearningMethods/
|-- ConAbl/
|-- ESD-SDXL/
|-- SculpMem/
`-- README.md
```

## ConAbl

[`ConAbl`](ConAbl/README.md) is the current unlearning method used across independent, sequential, and simultaneous CUIG experiments. It trains selected model parameters so target concepts are mapped toward anchor concepts, with optional anchor preservation and regularization.

See [`ConAbl/README.md`](ConAbl/README.md) for the method layout, direct training commands, supported concept inputs, anchor dataset behavior, trainable parameter groups, regularizer flags, and output conventions.

## SculpMem

[`SculpMem`](SculpMem/README.md) extends the ConAbl training interface with optional dynamic SelFT-style attention masking. It keeps the same anchor-based target-to-anchor objective, checkpoint format, concept inputs, and CUIG sampling/evaluation workflow, while allowing the active attention-weight subset to change during training.

See [`SculpMem/README.md`](SculpMem/README.md) for dynamic masking flags, direct training examples, regularizer compatibility, and output conventions.

## ESD-SDXL

[`ESD-SDXL`](ESD-SDXL/README.md) provides SDXL-oriented unlearning support used by the Celebrity workflows. It follows the same repository-level separation: method training code lives under `UnlearningMethods/ESD-SDXL`, reusable regularizers live under `Regularizers/`, and full experiment launchers live under `BashScripts/`.

## Adding a New Method

Add new methods as sibling directories under `UnlearningMethods/`.

Recommended structure:

- provide a method-specific training entrypoint
- keep argument parsing, data setup, model setup, and training utilities split under `src/`
- use regularizers from `Regularizers/` instead of duplicating shared logic
- write checkpoints under the user-provided `--output_dir`
- add BashScripts launchers for full experiment pipelines
- add a short method README if the method requires custom data or setup
