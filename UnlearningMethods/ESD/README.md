# ESD: Erasing Concepts from Stable Diffusion

This README provides instructions for training, sampling, and evaluating ESD (Erasing Stable Diffusion) models across independent, continual, and simultaneous configurations. It also outlines the use of ContinualEnhancement methods to improve unlearning performance over time.

This implementation builds on [Erasing Concepts from Diffusion Models by Rohit Gandikota et al](https://github.com/rohitgandikota/erasing).

---

---

## 🧠 Training
```bash
cd UnlearningMethods/ESD
```
### Independent
#### Style
```bash
python train-esd.py \
--erase_concept 'Abstractionism' \
--concept_type 'style' \
--train_method 'esd-x' \
--save_path $CUIG_ROOT/esd/models/independent/base/style/Abstractionism.safetensors \
--base_model_dir ../base_models/UnlearnCanvas
```
#### Object
#### Celebrity
### Continual
#### Style
#### Object
#### Celebrity
### Simultaneous
#### Style
#### Object
#### Celebrity

---

## 🔬 Evaluation

### 🔹 Sample Images
```bash
python sampling_unlearned_models/esd.py --ckpt /path/to/model.pth --output_dir /path/to/images ...
```

### 🔹 Evaluate Accuracy
```bash
python quantitative/accuracy.py --input_dir /path/to/images --output_dir /path/to/metrics ...
```

---

