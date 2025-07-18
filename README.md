# CUIG: Continual Unlearning for Image Generation

**CUIG** is a modular framework for developing, evaluating, and enhancing continual unlearning methods in image generation tasks. It supports plug-and-play integration of unlearning algorithms, evaluation metrics, and continual enhancement strategies.

---

## 📁 Project Structure

```
CUIG/
├── Analysis/              # Notebooks for analyzing results and performance
├── ContinualEnhancements/ # Methods to improve model retention over time
├── Evaluation/            # Evaluation metrics for unlearning and retention
├── UnlearningMethods/     # Unlearning algorithms and approaches
├── env.yaml               # Conda environment setup file
└── README.md              # Project documentation (this file)
```

---

## ⚙️ Setup

Use Conda to create and activate the project environment:

```bash
conda env create -f env.yaml
conda activate cuig
export CUIG_ROOT=/your/repo/dir/here
export OUTPUT_ROOT=/your/save/dir/here
export BASE_MODEL_DIR=/your/base/model/dir/here
export OPENAI_API_KEY=/your/openai/api/key/here
```

---

## 🚀 Usage

You can run different parts of the pipeline as follows:

- **Unlearning**: Add your methods in `UnlearningMethods/` and run them on pretrained diffusion models.
- **Evaluation**: Plug your evaluation metrics into the `Evaluation/` folder to test unlearning fidelity.
- **Enhancements**: Try retention methods in `ContinualEnhancements/`.

Each component is modular and can be used independently or as part of a larger continual unlearning pipeline.

---

## 📊 Evaluation Criteria

We evaluate unlearning methods using:

- **Unlearning Effectiveness**: Target forgetfulness (e.g., class, style, concept)
- **Retention Quality**: Preservation of non-target capabilities

---

## 🧪 Contributions & Experiments

If you're running or contributing experiments, organize your results and scripts under `Analysis/`. Please document new methods clearly and follow the folder structure.

---

## 📬 Contact

For questions, collaborations, or bugs, open an issue or contact [justinhylee135](https://github.com/justinhylee135).

---

## 📄 License

[MIT License](LICENSE) — open to academic and research use.
