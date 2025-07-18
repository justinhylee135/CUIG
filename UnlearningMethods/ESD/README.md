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
--train_method 'kv-xattn' \
--save_path $OUTPUT_ROOT/esd/models/independent/base/style/Abstractionism.pth \
--base_model_dir $BASE_MODEL_DIR 
```
#### Object
```bash
python train_esd.py \
--concept 'Bears' \
--concept_type 'object' \
--train_method 'kv-xattn' \
--save_path $OUTPUT_ROOT/esd/models/independent/base/object/Bears_test.pth \
--base_model_dir $BASE_MODEL_DIR \
--start_guidance 9.0 \
--negative_guidance 9.0
```

### Continual
Use the same arguments as independent but now add "--unet_ckpt" with the pass to your previous unlearned models .pth

### Simultaneous
#### Style
```bash
python train_esd.py \
--concept '["Abstractionism", "Byzantine", "Cartoon"]' \
--concept_type 'style' \
--train_method 'kv-xattn' \
--save_path $OUTPUT_ROOT/esd/models/simultaneous/base/style/thruCartoon/thruCartoon.pth \
--base_model_dir $BASE_MODEL_DIR \
--eval_every 100 \
--classifier_dir $REPO_ROOT/Evaluation/UnlearnCanvas/classifiers
```

### Regularization: L1
Same arguments as continual with with "--l1sp_weight" and add the scaling factor for l1sp. Good starting values are [1,100]

### Regularization: L2
Same arguments as continual with with "--l2sp_weight" and add the scaling factor for l2sp. Good starting values are [10000,300000]

### Selective Finetuning (SelFT)
Same arguments as continual but now we add the loss we want to use for one forward pass "selft_loss". The number of parameters to update "selft_topk" and where to save our gradient dictionary "selft_grad_dict_path"
#### Style
```bash
python train_esd.py \
--concept 'Abstractionism' \
--concept_type 'style' \
--train_method 'kv-xattn' \
--save_path $OUTPUT_ROOT/esd/models/continual/selft/style/0.01/thruAbstractionism.pth \
--base_model_dir $BASE_MODEL_DIR \
--selft_loss 'esd' \
--selft_topk 0.01 \
--selft_grad_dict_path $OUTPUT_ROOT/esd/models/continual/selft/style/0.01/thruAbstractionism_grad_dict.pth
```

### Gradient Projection
Same arguments as continual but now we add the flag "with_gradient_projection" and the prompts we want to build the text embedding subspace with or where to save the generated ones at "gradient_projection_prompts". The example below shows the first unlearning step, to unlearn the next one and preserve previous unlearning, add the previous unlearned concepts as a list to "--previously_unlearned".
#### Style
```bash
python train_esd.py \
--concept 'Abstractionism' \
--concept_type 'style' \
--train_method 'kv-xattn' \
--save_path $OUTPUT_ROOT/esd/models/continual/projection/gradient_projection/style/thruAbstractionism.pth \
--base_model_dir $BASE_MODEL_DIR \
--with_gradient_projection 
--gradient_projection_prompts $OUTPUT_ROOT/esd/models/continual/projection/gradient_projection/style/thruAbstractionism_prompts.txt

```

---

