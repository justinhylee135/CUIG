# CA: Concept Ablation

This README provides instructions for training CA (Concept Ablation) models across independent, continual, and simultaneous configurations. It also outlines the use of ContinualEnhancement methods to improve unlearning performance over time.

This implementation builds on [Ablating Concepts in Text-to-Image Diffusion Models by Nupur Kumari et al](https://github.com/nupurkmr9/concept-ablation/tree/main).

---

---

## 🧠 Training
```bash
cd $REPO_ROOT/UnlearningMethods/DoCo
```
### Independent
#### Style
```bash
accelerate launch \
    --config_file $REPO_ROOT/UnlearningMethods/DoCo/accelerate_config.yaml \
    train_doco.py \
    --caption_target "Cartoon" \
    --concept_type style \
    --output_dir $OUTPUT_ROOT/doco/models/independent/base/style/Cartoon \
    --base_model_dir $BASE_MODEL_DIR \
    --train_size 200 \
    --class_data_dir $REPO_ROOT/UnlearningMethods/DoCo/anchor_datasets/style/doco_painting_sd \
    --class_prompt $REPO_ROOT/UnlearningMethods/DoCo/anchor_prompts/style/doco_painting.txt  \
    --enable_xformers_memory_efficient_attention \
    --overwrite_existing_ckpt \
    --dlr 5e-4 \
    --discrim_num_warmup_steps 500 \
    --max_train_steps 1000 \
    --learning_rate 5e-5 \
    --hflip \
    --noaug \
    --gradient_clip \
    --with_prior_preservation \
    --discrim_model patchgan \
    --num_timesteps 50 \
    --use_random_latent

    --discrim_pixel_space
    --scale_lr
    --train_batch_size 4
```

#### Object
```bash
accelerate launch \
    --config_file $REPO_ROOT/UnlearningMethods/DoCo/accelerate_config.yaml \
    train_doco.py \
    --caption_target "horse+bear" \
    --concept_type object \
    --output_dir $OUTPUT_ROOT/doco/models/independent/base/object/Bears \
    --base_model_dir $REPO_ROOT/UnlearningMethods/base_models/UnlearnCanvas  \
    --num_class_images 200 \
    --class_data_dir $REPO_ROOT/UnlearningMethods/DoCo/anchor_datasets/object/Horses \
    --class_prompt $REPO_ROOT/UnlearningMethods/DoCo/anchor_prompts/object/Horses.txt  \
    --enable_xformers_memory_efficient_attention \
    --overwrite_existing_ckpt \
    --dlr 5e-4 \
    --discrim_num_warmup_steps 500 \
    --max_train_steps 1000 \
    --learning_rate 5e-5 \
    --hflip \
    --noaug \
    --gradient_clip \
    --with_prior_preservation \
    --discrim_model patchgan \
    --num_timesteps 50 \
    --use_random_latent
    
```
```bash
accelerate launch \
    --config_file $REPO_ROOT/UnlearningMethods/DoCo/accelerate_config.yaml \
    train_doco.py \
    --caption_target '["architecture+bear", "butterfly+bear", "flame+bear", "flower+bear", "horse+bear", "human+bear", "sea+bear", "tree+bear"]' \
    --concept_type object \
    --output_dir $OUTPUT_ROOT/doco/models/independent/base/object_all/Bears \
    --base_model_dir $REPO_ROOT/UnlearningMethods/base_models/UnlearnCanvas  \
    --max_train_steps 2000 \
    --num_class_images 200 \
    --class_data_dir '["/users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/DoCo/anchor_datasets/object/Architectures",
                        "/users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/DoCo/anchor_datasets/object/Butterfly",
                        "/users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/DoCo/anchor_datasets/object/Flame",
                        "/users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/DoCo/anchor_datasets/object/Flowers",
                        "/users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/DoCo/anchor_datasets/object/Horses",
                        "/users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/DoCo/anchor_datasets/object/Human",
                        "/users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/DoCo/anchor_datasets/object/Sea",
                        "/users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/DoCo/anchor_datasets/object/Trees"]' \
    --class_prompt '["/users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/DoCo/anchor_prompts/object/Architectures.txt",
                     "/users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/DoCo/anchor_prompts/object/Butterfly.txt",
                     "/users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/DoCo/anchor_prompts/object/Flame.txt",
                     "/users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/DoCo/anchor_prompts/object/Flowers.txt",
                     "/users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/DoCo/anchor_prompts/object/Horses.txt",
                     "/users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/DoCo/anchor_prompts/object/Human.txt",
                     "/users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/DoCo/anchor_prompts/object/Sea.txt",
                     "/users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/DoCo/anchor_prompts/object/Trees.txt"]'  \
    --scale_lr --hflip --noaug \
    --enable_xformers_memory_efficient_attention 
```

### Continual
Simply use the same arguments as independent but add the argument "--unet_ckpt" and the path to your previously unlearned model's delta.bin.

### Simultaneous
For simultaneous now we pass in "caption_target" as a list. Additionally we add eval_every and classifier_dir for early stopping. We set the upperbound train steps to 6000 and the patience to 2000.
#### Style
```bash
accelerate launch \
    --config_file $REPO_ROOT/UnlearningMethods/DoCo/accelerate_config.yaml \
    train_doco.py \
    --caption_target '["Abstractionism", "Byzantine"]' \
    --concept_type style \
    --output_dir $OUTPUT_ROOT/doco/models/simultaneous/base/early_stopping/style/thruByzantine \
    --base_model_dir $REPO_ROOT/UnlearningMethods/base_models/UnlearnCanvas  \
    --train_size 200 \
    --scale_lr --hflip --noaug \
    --enable_xformers_memory_efficient_attention \
    --class_data_dir $REPO_ROOT/UnlearningMethods/DoCo/anchor_datasets/style/painting_unlearncanvas_diffusers \
    --class_prompt $REPO_ROOT/UnlearningMethods/DoCo/anchor_prompts/style/painting.txt \
    --max_train_steps 6000 \
    --patience 2000 \
    --eval_every 100 \
    --classifier_dir $REPO_ROOT/Evaluation/UnlearnCanvas/classifiers
```

### Regularization: L1
Same arguments as continual with with "--l1sp_weight" and add the scaling factor for l1sp. Good starting values are [1,100]


### Regularization: L2
Same arguments as continual with with "--l2sp_weight" and add the scaling factor for l2sp. Good starting values are [10000,300000]



### Selective Finetuning (SelFT)
Same arguments as continual but now we add the loss we want to do one forward pass of "selft_loss" and the percentage of parameters we want to actually update "selft_topk". We save the gradient dictionary to use again in "selft_grad_dict_path"
#### Style
```bash
accelerate launch \
    --config_file $REPO_ROOT/UnlearningMethods/DoCo/accelerate_config.yaml \
    train_doco.py \
    --caption_target "Abstractionism" \
    --concept_type style \
    --output_dir $OUTPUT_ROOT/doco/models/continual/selft/style/0.05/thruAbstractionism \
    --base_model_dir $BASE_MODEL_DIR  \
    --max_train_steps 1000 \
    --train_size 200 \
    --class_data_dir $REPO_ROOT/UnlearningMethods/DoCo/anchor_datasets/style/painting_unlearncanvas_diffusers \
    --class_prompt $REPO_ROOT/UnlearningMethods/DoCo/anchor_prompts/style/painting.txt  \
    --scale_lr --hflip --noaug \
    --enable_xformers_memory_efficient_attention \
    --selft_loss "ca" \
    --selft_topk 0.05 \
    --selft_grad_dict_path $OUTPUT_ROOT/doco/models/continual/selft/style/thruAbstractionism/0.05/grad_dict.pt
```

### Gradient Projection
Same arguments as continual but now we add the flag "with_gradient_projection" and the prompts we want to build the text embedding subspace with or where to save the generated ones at "gradient_projection_prompts". The example below shows the first unlearning step, to unlearn the next one and preserve previous unlearning, add the previous unlearned concepts as a list to "--previously_unlearned".
#### Style
```bash
accelerate launch \
--config_file $REPO_ROOT/UnlearningMethods/DoCo/accelerate_config.yaml \
    train_doco.py \
    --caption_target "Abstractionism" \
    --concept_type style \
    --output_dir $OUTPUT_ROOT/doco/models/continual/projection/style/gradient_projection/thruAbstractionism \
    --base_model_dir $REPO_ROOT/UnlearningMethods/base_models/UnlearnCanvas  \
    --max_train_steps 1000 \
    --train_size 200 \
    --class_data_dir $REPO_ROOT/UnlearningMethods/DoCo/anchor_datasets/style/painting_unlearncanvas_diffusers \
    --class_prompt $REPO_ROOT/UnlearningMethods/DoCo/anchor_prompts/style/painting.txt  \
    --scale_lr --hflip --noaug \
    --enable_xformers_memory_efficient_attention \
    --with_gradient_projection \
    --gradient_projection_prompts "$OUTPUT_ROOT/doco/models/continual/projection/style/gradient_projection/thruAbstractionism/prompts.txt"
```
---

