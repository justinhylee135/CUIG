# ESD: Erasing Concepts from Stable Diffusion

This README provides instructions for training ESD (Erasing Stable Diffusion) models across independent, continual, and simultaneous configurations. It also outlines the use of ContinualEnhancement methods to improve unlearning performance over time.

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
```bash
python train_esd.py \
--concept 'Abstractionism' \
--concept_type 'style' \
--train_method 'kv-xattn' \
--save_path $OUTPUT_ROOT/esd/models/continual/projection/gradient_projection/style/thruAbstractionism.pth \
--base_model_dir $BASE_MODEL_DIR \
--with_gradient_projection \
--gradient_projection_prompts $OUTPUT_ROOT/esd/models/continual/projection/gradient_projection/style/thruAbstractionism_prompts.txt
```
#### Object
```bash
python -m debugpy --listen 10.6.8.8:5678 --wait-for-client \
python train_esd.py \
--concept 'Bears' \
--concept_type 'object' \
--train_method 'esd-u' \
--save_path $OUTPUT_ROOT/esd/models/independent/base/object/Bears_test.pth \
--base_model_dir $BASE_MODEL_DIR \
--overwrite_existing_ckpt 
```
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
#### Object
#### Celebrity

### Regularization: L1
#### Style
```bash
python train_esd.py \
--concept 'Abstractionism' \
--concept_type 'style' \
--train_method 'xattn' \
--save_path $OUTPUT_ROOT/esd/models/continual/l1sp/style/100/thruAbstractionism.pth \
--base_model_dir $BASE_MODEL_DIR \
--l1sp_weight 100
```

### Regularization: L2
#### Style
```bash
python train_esd.py \
--concept 'Abstractionism' \
--concept_type 'style' \
--train_method 'xattn' \
--save_path $OUTPUT_ROOT/esd/models/continual/l1sp/style/300000/thruAbstractionism.pth \
--base_model_dir $BASE_MODEL_DIR \
--l1sp_weight 300000
```

### Selective Finetuning (SelFT)
#### Style
```bash
python train_esd.py \
--concept 'Abstractionism' \
--concept_type 'style' \
--train_method 'xattn' \
--save_path $OUTPUT_ROOT/esd/models/continual/selft/style/0.01/thruAbstractionism.pth \
--base_model_dir $BASE_MODEL_DIR \
--selft_loss 'esd' \
--selft_topk 0.01 \
--selft_grad_dict_path $OUTPUT_ROOT/esd/models/continual/selft/style/0.01/thruAbstractionism_grad_dict.pth
```

---

