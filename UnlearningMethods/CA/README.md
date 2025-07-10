# CA: Concept Ablation

This README provides instructions for training CA (Concept Ablation) models across independent, continual, and simultaneous configurations. It also outlines the use of ContinualEnhancement methods to improve unlearning performance over time.

This implementation builds on [Ablating Concepts in Text-to-Image Diffusion Models by Nupur Kumari et al](https://github.com/nupurkmr9/concept-ablation/tree/main).

---

---

## 🧠 Training
```bash
cd $REPO_ROOT/UnlearningMethods/CA
```
### Independent
#### Style
```bash
accelerate launch \
    --config_file $REPO_ROOT/UnlearningMethods/CA/accelerate_config.yaml \
python -m debugpy --listen 10.6.5.1:5678 --wait-for-client \
    train_ca.py \
    --caption_target "Abstractionism" \
    --concept_type style \
    --output_dir $OUTPUT_ROOT/ca/models/independent/base/style/Abstractionism_retention \
    --base_model_dir $BASE_MODEL_DIR  \
    --max_train_steps 1000 \
    --train_size 200 \
    --class_data_dir $REPO_ROOT/UnlearningMethods/CA/anchor_datasets/style/painting_unlearncanvas_diffusers_abstracitonism \
    --class_prompt $REPO_ROOT/UnlearningMethods/CA/anchor_prompts/style/painting.txt  \
    --scale_lr --hflip --noaug \
    --enable_xformers_memory_efficient_attention \
    --with_prior_preservation \
    --with_negative_prompt
```
```bash
accelerate launch \
    --config_file $REPO_ROOT/UnlearningMethods/CA/accelerate_config.yaml \
    train_ca.py \
    --caption_target "Abstractionism" \
    --concept_type style \
    --output_dir $OUTPUT_ROOT/ca/models/independent/gradient_projection/style/Abstractionism \
    --base_model_dir $BASE_MODEL_DIR  \
    --max_train_steps 1000 \
    --num_class_images 200 \
    --class_data_dir $REPO_ROOT/UnlearningMethods/CA/anchor_datasets/style/painting_unlearncanvas_diffusers \
    --class_prompt $REPO_ROOT/UnlearningMethods/CA/anchor_prompts/style/painting.txt  \
    --scale_lr --hflip --noaug \
    --enable_xformers_memory_efficient_attention \
    --with_gradient_projection
    # --with_prior_preservation 
```
#### Object
```bash
accelerate launch \
    --config_file $REPO_ROOT/UnlearningMethods/CA/accelerate_config.yaml \
python -m debugpy --listen 10.6.5.4:5678 --wait-for-client \
    train_ca.py \
    --caption_target "*+An image of Bears" \
    --concept_type object \
    --output_dir /fs/scratch/PAS2099/lee.10369/CUIG/ca/models/independent/base/object/Bears_test \
    --base_model_dir /users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/base_models/UnlearnCanvas  \
    --max_train_steps 2000 \
    --num_class_images 200 \
    --class_data_dir /users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/CA/anchor_datasets/object/Horses \
    --class_prompt /users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/CA/anchor_prompts/object/Horses.txt  \
    --scale_lr --hflip --noaug \
    --enable_xformers_memory_efficient_attention \
    --with_prior_preservation \
    --overwrite_existing_ckpt 
```
#### Celebrity

### Continual
```bash
accelerate launch \
--config_file /users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/CA/accelerate_config.yaml \
train_ca.py \
    --caption_target "Byzantine Style" \
    --concept_type style \
    --output_dir /fs/scratch/PAS2099/lee.10369/CUIG/ca/models/continual/base/style/thruByzantine \
    --unet_ckpt /fs/scratch/PAS2099/lee.10369/CUIG/ca/models/independent/base/style/Abstractionism/delta.bin \
    --base_model_dir /users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/base_models/UnlearnCanvas  \
    --max_train_steps 1000 \
    --train_size 200 \
    --class_data_dir /users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/CA/anchor_datasets/style/painting_unlearncanvas \
    --class_prompt /users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/CA/anchor_prompts/style/painting.txt  \
    --scale_lr --hflip --noaug \
    --enable_xformers_memory_efficient_attention \
    --with_prior_preservation 
```
#### Style
#### Object
#### Celebrity

### Simultaneous
#### Style
```bash
accelerate launch \
    --config_file $REPO_ROOT/UnlearningMethods/CA/accelerate_config.yaml \
    train_ca.py \
    --caption_target '["Abstractionism", "Byzantine"]' \
    --concept_type style \
    --output_dir $OUTPUT_ROOT/ca/models/simultaneous/base/style/thruByzantine \
    --base_model_dir $BASE_MODEL_DIR  \
    --max_train_steps 1000 \
    --train_size 200 \
    --class_data_dir $REPO_ROOT/UnlearningMethods/CA/anchor_datasets/style/painting_unlearncanvas_diffusers \
    --class_prompt $REPO_ROOT/UnlearningMethods/CA/anchor_prompts/style/painting.txt  \
    --scale_lr --hflip --noaug \
    --enable_xformers_memory_efficient_attention \
    --with_prior_preservation \
    --eval_every 100 \
    --classifier_dir $REPO_ROOT/Evaluation/UnlearnCanvas/classifiers
```
#### Object
#### Celebrity

### Regularization: L1
#### Style


### Regularization: L2
#### Style


### Selective Finetuning (SelFT)
#### Style
```bash
accelerate launch \
    --config_file $REPO_ROOT/UnlearningMethods/CA/accelerate_config.yaml \
    train_ca.py \
    --caption_target "Abstractionism" \
    --concept_type style \
    --output_dir $OUTPUT_ROOT/ca/models/continual/selft/style/thruAbstractionism \
    --base_model_dir $BASE_MODEL_DIR  \
    --max_train_steps 1000 \
    --train_size 200 \
    --class_data_dir $REPO_ROOT/UnlearningMethods/CA/anchor_datasets/style/painting_unlearncanvas_diffusers \
    --class_prompt $REPO_ROOT/UnlearningMethods/CA/anchor_prompts/style/painting.txt  \
    --scale_lr --hflip --noaug \
    --enable_xformers_memory_efficient_attention \
    --selft_loss "ca" \
    --selft_topk 0.01 \
    --selft_grad_dict_path $OUTPUT_ROOT/ca/models/continual/selft/style/thruAbstractionism/grad_dict.pt
```

---

