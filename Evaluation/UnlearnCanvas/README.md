## 🔬 Evaluation

Follow these steps to sample from a model and evaluate its performance after unlearning specific concepts.

### 1. Change Directory

Navigate to the evaluation folder:

```bash
cd $REPO_ROOT/Evaluation/UnlearnCanvas
```

### 2. Sample Images

Example: sampling from a model that has sequentially unlearned the styles **Abstractionism**, **Byzantine**, and **Cartoon** using Gradient Projection and CA:

```bash
python sample.py \
    --unet_ckpt_path $OUTPUT_ROOT/ca/models/continual/projection/style/gradient_projection/thruCartoon/delta.bin \
    --output_dir $OUTPUT_ROOT/ca/eval_results/continual/projection/style/gradient_projection/thruCartoon/images \
    --styles_subset '["Abstractionism", "Byzantine", "Cartoon", "Blossom_Season", "Rust", "Crayon", "Fauvism", "Superstring", "Red_Blue_Ink", "Gorgeous_Love", "French", "Joy", "Greenfield", "Expressionism", "Impressionism"]' \
    --objects_subset '["Architectures", "Butterfly", "Flame", "Flowers", "Horses", "Human", "Sea", "Trees"]' \
    --pipeline_dir $BASE_MODEL_DIR 
```

### 3. Evaluate Accuracy

Run the evaluator on the sampled images:

```bash
python evaluate.py \
    --input_dir $OUTPUT_ROOT/ca/eval_results/continual/projection/style/gradient_projection/thruCartoon/images \
    --output_dir $OUTPUT_ROOT/ca/eval_results/continual/projection/style/gradient_projection/thruCartoon/metrics \
    --classifier_dir $REPO_ROOT/Evaluation/UnlearnCanvas/classifiers \
    --unlearn '["Abstractionism", "Byzantine", "Cartoon"]' \
    --retain '["Blossom_Season", "Rust", "Crayon", "Fauvism", "Superstring", "Red_Blue_Ink", "Gorgeous_Love", "French", "Joy", "Greenfield", "Expressionism", "Impressionism"]' \
    --cross_retain '["Architectures", "Butterfly", "Flame", "Flowers", "Horses", "Human", "Sea", "Trees"]'
```

**Notes:**
- Adjust the paths and style/object lists as needed for your experiments.
- Ensure all required dependencies are installed before running the scripts.
