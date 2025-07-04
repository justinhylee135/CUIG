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
    train_ca.py \
    --caption_target "Abstractionism Style" \
    --concept_type style \
    --output_dir $OUTPUT_ROOT/ca/models/independent/base/style/Abstractionism_retention \
    --base_model_dir $BASE_MODEL_DIR  \
    --max_train_steps 1000 \
    --train_size 200 \
    --class_data_dir $REPO_ROOT/UnlearningMethods/CA/anchor_datasets/style/painting_unlearncanvas \
    --class_prompt $REPO_ROOT/UnlearningMethods/CA/anchor_prompts/style/painting.txt  \
    --scale_lr --hflip --noaug \
    --enable_xformers_memory_efficient_attention \
    --with_prior_preservation 
```

#### Object
```bash
accelerate launch \
    --config_file $REPO_ROOT/UnlearningMethods/CA/accelerate_config.yaml \
    --config_file /users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/CA/accelerate_config.yaml \
python -m debugpy --listen 10.6.7.11:5678 --wait-for-client \
    train_ca.py \
    --caption_target "*+An image of Bears" \
    --concept_type object \
    --output_dir /fs/scratch/PAS2099/lee.10369/CUIG/ca/models/independent/base/object/Bears \
    --base_model_dir /users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/base_models/UnlearnCanvas  \
    --max_train_steps 2000 \
    --train_size 200 \
    --class_data_dir /users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/CA/anchor_datasets/object/Horses \
    --class_prompt /users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/CA/anchor_prompts/object/Horses.txt  \
    --scale_lr --hflip --noaug \
    --enable_xformers_memory_efficient_attention \
    --with_prior_preservation 
```
#### Celebrity

### Continual
#### Style
#### Object
#### Celebrity

### Simultaneous
#### Style

#### Object
#### Celebrity

### Regularization: L1
#### Style


### Regularization: L2
#### Style


### Selective Finetuning (SelFT)
#### Style


---

