# ESD: Erasing Concepts from Stable Diffusion

This README provides instructions for training, sampling, and evaluating ESD (Erasing Stable Diffusion) models across independent, continual, and simultaneous configurations. It also outlines the use of ContinualEnhancement methods to improve unlearning performance over time.

This implementation builds on [Erasing Concepts from Diffusion Models by Rohit Gandikota et al](https://github.com/rohitgandikota/erasing).

---

---

## 🧠 Training
```bash
cd $REPO_ROOT/UnlearningMethods/ESD
```
### Independent
#### Style
```bash
python train_esd.py \
--concept 'Abstractionism' \
--concept_type 'style' \
--train_method 'xattn' \
--save_path $OUTPUT_ROOT/esd/models/independent/base/style/Abstractionism.pth \
--base_model_dir $BASE_MODEL_DIR 
```

#### Object
#### Celebrity
### Continual
#### Style
#### Object
#### Celebrity
### Simultaneous
#### Style
```bash
python train_esd.py \
--concept '["Abstractionism", "Byzantine", "Cartoon"]' \
--concept_type 'style' \
--train_method 'xattn' \
--save_path $OUTPUT_ROOT/esd/models/simultaneous/base/style/thruCartoon/thruCartoon.pth \
--base_model_dir $BASE_MODEL_DIR \
--eval_every 100 \
--classifier_dir $REPO_ROOT/Evaluation/UnlearnCanvas/classifiers
```
#### Style
#### Object
#### Celebrity

---

