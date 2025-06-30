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
python train.py -t \
    --gpus 0, \
    --concept_type style \
    --logdir /fs/scratch/PAS2099/lee.10369/mmuc_results/ca/models/debug/simultaneous/thruByzantine \
    --name "ca_thruByzantine"  \
    --caption_target  '["Abstractionism Style", "Byzantine Style"]' \
    --train_max_steps 100 \
    --train_size 200 \
    --resume_from_checkpoint_custom /users/PAS2099/justinhylee135/Research/UnlearningDM/MMDU/external_model_ckpts/UnlearnCanvas/unlearncanvas_compvis.ckpt \
    --prompts /users/PAS2099/justinhylee135/Research/UnlearningDM/MMUC/machine_unlearning/mu_concept_ablation_ca/anchor_datasets/doco_unlearncanvas_samples_painting/prompts.txt \
    --root /users/PAS2099/justinhylee135/Research/UnlearningDM/MMUC/machine_unlearning/mu_concept_ablation_ca/anchor_datasets/doco_unlearncanvas_samples_painting 
```
```bash
accelerate launch \
    --config_file $REPO_ROOT/UnlearningMethods/CA/accelerate_config.yaml \
    train_ca.py \
    --pretrained_model_name_or_path=$BASE_MODEL_DIR  \
    --output_dir=$OUTPUT_ROOT/ca/models/independent/base/style/Abstractionism \
    --class_data_dir=./data/samples_painting/ \
    --class_prompt="painting"  \
    --caption_target "Abstractionism" \
    --concept_type style \
    --resolution=512  \
    --train_batch_size=4  \
    --learning_rate=2e-6  \
    --max_train_steps=100 \
    --scale_lr --hflip --noaug \
    --parameter_group cross-attn \
    --enable_xformers_memory_efficient_attention 
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

### Regularization: L1
#### Style


### Regularization: L2
#### Style


### Selective Finetuning (SelFT)
#### Style


---

