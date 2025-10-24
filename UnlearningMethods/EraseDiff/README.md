# Collect Datasets
```bash
python $REPO_ROOT/UnlearningMethods/EraseDiff/UnlearnCanvas_generate_dataset.py
cd $REPO_ROOT/UnlearningMethods/EraseDiff
```

# Train style removal with diffusers
```bash
python train_erasediff.py \
    --concept "Dogs" \
    --concept_type "object" \
    --anchor "Horses" \
    --base_model_dir "/users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/base_models/UnlearnCanvas" \
    --train_method "xattn" \
    --lambda_bome 1.0 \
    --cycles 1 \
    --steps_per_cycle 100 \
    --K_steps 1 \
    --output_path "/fs/scratch/PAS2099/lee.10369/CUIG/erasediff/models/independent/base/object/Dogs.pth" \
    --lr 5e-5
```

```bash
python train_erasediff.py \
    --concept '["bear", "bird"]' \
    --anchor '["horse", "butterfly"]' \
    --concept_type "object" \
    --base_model_dir "/users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/base_models/UnlearnCanvas" \
    --train_method "xattn" \
    --lambda_bome 1.0 \
    --cycles 1 \
    --steps_per_cycle 100 \
    --K_steps 1 \
    --output_path "/fs/scratch/PAS2099/lee.10369/CUIG/erasediff/models/independent/base/object/Dogs.pth" \
    --data_method 'ca' \
    --forget_data_dir '["/users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/CA/anchor_datasets/object/Horses", "/users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/CA/anchor_datasets/object/Butterfly"]' \
    --remain_data_dir '["/users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/CA/anchor_datasets/object/Horses", "/users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/CA/anchor_datasets/object/Butterfly"]' \
    --remain_weight 0.005 \
    --lr 2e-6 \
    --overwrite_existing_ckpt
```


```bash
python train_erasediff.py \
    --concept "bear" \
    --anchor "horse" \
    --concept_type "object" \
    --data_method "ca" \
    --forget_data_dir "/users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/CA/anchor_datasets/object/Horses" \
    --remain_data_dir "/users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/CA/anchor_datasets/object/Horses" \
    --base_model_dir "/users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/base_models/UnlearnCanvas" \
    --train_method "xattn" \
    --lambda_bome 1.0 \
    --cycles 5 \
    --steps_per_cycle 100 \
    --K_steps 2 \
    --output_path "/fs/scratch/PAS2099/lee.10369/CUIG/erasediff/models/independent/base/object/Bears_datamethodCA.pth" \
    --lr 5e-5 \
    --overwrite_existing_ckpt
```

```bash
python train_erasediff.py \
    --concept "Abstractionism" \
    --anchor "Seed_Images" \
    --concept_type "style" \
    --data_method "ca" \
    --forget_data_dir "/users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/CA/anchor_datasets/style/painting_unlearncanvas_diffusers" \
    --remain_data_dir "/users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/CA/anchor_datasets/style/painting_unlearncanvas_diffusers" \
    --base_model_dir "/users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/base_models/UnlearnCanvas" \
    --train_method "xattn" \
    --lambda_bome 1.00 \
    --K_steps 1 \
    --cycles 10 \
    --steps_per_cycle 100 \
    --lr 5e-7 \
    --output_path "/fs/scratch/PAS2099/lee.10369/CUIG/erasediff/models/independent/base/style/Abstractionism_K1_Cycles10_Steps100_LR5e7_l1sp10.pth" \
    --overwrite_existing_ckpt \
    --l1sp_weight 10
```

```bash
python train_erasediff.py \
    --concept "Byzantine" \
    --anchor "Seed_Images" \
    --concept_type "style" \
    --data_method "ca" \
    --forget_data_dir "/users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/CA/anchor_datasets/style/painting_unlearncanvas_diffusers" \
    --remain_data_dir "/users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/CA/anchor_datasets/style/painting_unlearncanvas_diffusers" \
    --base_model_dir "/users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/base_models/UnlearnCanvas" \
    --unet_ckpt "/fs/scratch/PAS2099/lee.10369/CUIG/erasediff/models/independent/base/style/Abstractionism_K1_Cycles10_Steps100_LR5e7.pth" \
    --train_method "xattn" \
    --lambda_bome 1.00 \
    --K_steps 1 \
    --cycles 10 \
    --steps_per_cycle 100 \
    --lr 5e-7 \
    --output_path "/fs/scratch/PAS2099/lee.10369/CUIG/erasediff/models/independent/base/style/Byzantine_K1_Cycles10_Steps100_LR5e7.pth" \
    --overwrite_existing_ckpt \
    --verbose
```
