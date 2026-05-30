<div align="center">

# CUIG

### Continual Unlearning for Image Generation

[![GitHub](https://img.shields.io/badge/GitHub-CUIG-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/justinhylee135/CUIG)
[![Paper](https://img.shields.io/badge/Paper-arXiv-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2511.07970)
[![Project Page](https://img.shields.io/badge/Project_Page-CUIG-2B6CB0?style=for-the-badge)](https://justinhylee135.github.io/CUIG_Project_Page/)

</div>

CUIG is a research codebase for studying concept unlearning in text-to-image diffusion models. The repository organizes unlearning methods, regularizers, evaluation suites, and end-to-end experiment scripts for independent, sequential, and simultaneous unlearning settings.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Documentation Map](#documentation-map)
- [Setup](#setup)
- [Running Experiments](#running-experiments)
- [Extending the Repo](#extending-the-repo)
- [To-Do](#to-do)
- [Citation](#citation)
- [License](#license)

## Overview

CUIG separates the unlearning method from regularizers so that the same regularizers can be reused across different unlearning methods.

The main experiment settings are:

- **Independent unlearning**: unlearn one concept from the base model for baseline comparisons.
- **Sequential unlearning**: unlearn concepts one after another, starting each request from the previously unlearned model.
- **Simultaneous unlearning**: unlearn cumulative concept sets from the base model.

## Repository Structure

```text
CUIG/
|-- BashScripts/              # End-to-end experiment entrypoints
|-- Checkpoints/              # Local evaluation classifier and image generator checkpoints
|-- Configs/                  # Accelerate configuration files
|-- Evaluation/               # Sampling and evaluation suites
|-- Regularizers/             # Reusable continual-unlearning regularizers
|-- UnlearningMethods/        # Unlearning method implementations
|-- env.yaml                  # Conda environment
`-- README.md
```

## Documentation Map

For detailed setup and usage, use the README closest to the code you are working with:

- [`BashScripts/README.md`](BashScripts/README.md): public-friendly configuration and experiment launch flow
- [`Checkpoints/README.md`](Checkpoints/README.md): expected checkpoint layout and download notes
- [`Configs/README.md`](Configs/README.md): Accelerate configuration files
- [`Evaluation/README.md`](Evaluation/README.md): benchmark evaluation overview with links to each benchmark README
- [`Regularizers/README.md`](Regularizers/README.md): Weight, Projection, SelFT, Merge, and Simultaneous helpers
- [`UnlearningMethods/README.md`](UnlearningMethods/README.md): method index
- [`UnlearningMethods/ConAbl/README.md`](UnlearningMethods/ConAbl/README.md): ConAbl-specific training details

## Setup

Create the conda environment:

```bash
conda env create -f env.yaml
conda activate cuig
```

For script configuration, copy and edit the example config described in [`BashScripts/README.md`](BashScripts/README.md). For model and evaluator assets, see [`Checkpoints/README.md`](Checkpoints/README.md).

## Running Experiments

Most experiments are script-oriented. Start with [`BashScripts/README.md`](BashScripts/README.md), then use the relevant script under `BashScripts/Independent`, `BashScripts/Sequential`, or `BashScripts/Simultaneous`.

The scripts generally handle training, sampling, and evaluation. For direct method usage, see [`UnlearningMethods/ConAbl/README.md`](UnlearningMethods/ConAbl/README.md). For benchmark-specific evaluation commands, see [`Evaluation/README.md`](Evaluation/README.md).

## Extending the Repo

Use the existing layout when adding new functionality:

- Add new unlearning methods under `UnlearningMethods/`.
- Add reusable continual-unlearning regularizers under `Regularizers/`.
- Add new sampling/evaluation code under `Evaluation/`.
- Add complete train/sample/evaluate pipelines under `BashScripts/`.
- Keep benchmark assets, downloaded classifiers, and generator checkpoints under `Checkpoints/`.

When adding a method, prefer exposing method-specific options through its training script while keeping reusable preservation logic in `Regularizers/`.

## To-Do

- Bash scripts for Celebrity ESD-SDXL
- Bash scripts for UnlearnCanvas SculpMem

## Citation

```bibtex
@inproceedings{lee2026continual,
  title={Continual Unlearning for Text-to-Image Diffusion Models: A Regularization Perspective},
  author={Lee, Justin and Mai, Zheda and Yoo, Jinsu and Fan, Chongyu and Zhang, Cheng and Chao, Wei-Lun},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## License

This repository is released under the MIT License. See `LICENSE` for details.
