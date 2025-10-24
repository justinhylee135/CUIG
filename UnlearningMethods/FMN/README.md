```bash
cd $REPO_ROOT/UnlearningMethods/FMN
```

```bash
accelerate launch \
    --config_file $REPO_ROOT/UnlearningMethods/FMN/accelerate_config.yaml \
    train_fmn.py \
    --caption_target "Abstractionism" \
    --concept_type style \
    --output_dir $OUTPUT_ROOT/fmn/models/independent/base/style/Abstractionism_test \
    --base_model_dir $BASE_MODEL_DIR \
    --max_train_steps 1000 \
    --train_size 200 \
    --class_data_dir "$REPO_ROOT/UnlearningMethods/FMN/anchor_datasets/style/Abstractionism_test" \
    --class_prompt "$REPO_ROOT/UnlearningMethods/FMN/anchor_prompts/style/Abstractionism_test.txt" \
    --hflip \
    --noaug \
    --enable_xformers_memory_efficient_attention \
    --overwrite_existing_ckpt
```

```bash
accelerate launch \
    --config_file $REPO_ROOT/UnlearningMethods/FMN/accelerate_config.yaml \
    train_fmn.py \
    --caption_target "Bears" \
    --concept_type "object" \
    --output_dir $OUTPUT_ROOT/fmn/models/independent/base/object/Bears \
    --base_model_dir $BASE_MODEL_DIR \
    --max_train_steps 100 \
    --train_size 200 \
    --class_data_dir "$REPO_ROOT/UnlearningMethods/FMN/anchor_datasets/object/Bears" \
    --class_prompt "$REPO_ROOT/UnlearningMethods/FMN/anchor_prompts/object/Bears.txt" \
    --hflip \
    --noaug \
    --enable_xformers_memory_efficient_attention \
    --overwrite_existing_ckpt
```