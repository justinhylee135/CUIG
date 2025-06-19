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
python train-esd.py \
--erase_concept 'Abstractionism Style' \
--concept_type 'style' \
--train_method 'esd-x' \
--save_path $OUTPUT_ROOT/esd/models/independent/base/style/Abstractionism.safetensors \
--base_model_dir $BASE_MODEL_DIR \
--iterations 200 \
--lr 0.00005 \
--negative_guidance 2 \
--torch_dtype bfloat16
```

python train-esd-original.py \
--erase_concept 'Abstractionism Style' \
--train_method 'esd-x' \
--save_path $OUTPUT_ROOT/esd/models/independent/base/style/Abstractionism.safetensors \
--base_model_dir $BASE_MODEL_DIR \
--iterations 200 

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

