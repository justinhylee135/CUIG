## 🔬 Evaluation

Follow these steps to sample from a model and evaluate its performance after unlearning a copyrighted Character

### 1. Change Directory

Navigate to the evaluation folder:

```bash
cd $REPO_ROOT/Evaluation/Character
conda activate cuig
```

### 2. Sample Images

Example: sampling from a model that has sequentially unlearned the keywords **Snoopy**, **Mickey Mouse**, **Spongebob**:

```bash
python sample_character.py \
    --model_name "SD" \
    --characters '["Iron_Man", "Homer_Simpson", "Batman", "Naruto_Uzumaki", "Avatar_Aang", "Hulk", "Captain_America", "Bart_Simpson", "Superman", "Sasuke_Uchiha", "Zuko", "Thor"]'\
    --output_dir "/fs/scratch/PAS2099/lee.10369/CUIG/stablediffusion/v1.4/character/images"
```

### 3. Evaluate Using Classifier

Setup
1. cd $REPO_ROOT/Evaluation/Character
2. wget https://raw.githubusercontent.com/Artanisax/Six-CD/main/ResNet/ckpts/resnet50_copyright_101_71.pt

Run the character classifier on the sampled images:

```bash
python evaluate_character.py \
    --input_dir "/fs/scratch/PAS2099/lee.10369/CUIG/stablediffusion/v1.4/character/images" \
    --output_dir "/fs/scratch/PAS2099/lee.10369/CUIG/stablediffusion/v1.4/character/metrics" \
    --unlearn '["Iron_Man", "Homer_Simpson", "Batman", "Naruto_Uzumaki", "Avatar_Aang", "Hulk"]' \
    --retain '["Captain_America", "Bart_Simpson", "Superman", "Sasuke_Uchiha", "Zuko", "Thor"]'
```

accelerate launch \
    --config_file /users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/CA/accelerate_config.yaml \
    train_ca.py \
    --caption_target "man+Iron Man" \
    --concept_type character \
    --output_dir "/fs/scratch/PAS2099/lee.10369/CUIG/ca/models/continual/base/character/steps2000_bsz4/thruIron_Man" \
    --base_model_dir "CompVis/stable-diffusion-v1-4"  \
    --max_train_steps 2000 \
    --num_class_images 200 \
    --class_data_dir /users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/CA/anchor_datasets/character/man \
    --class_prompt /users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/CA/anchor_prompts/character/man.txt \
    --scale_lr --hflip --noaug \
    --enable_xformers_memory_efficient_attention 

**Notes:**
- Adjust the paths and style/object lists as needed for your experiments.
- Ensure all required dependencies are installed before running the scripts.
