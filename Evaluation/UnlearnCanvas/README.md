## 🔬 Evaluation

Follow these steps to sample from a model and evaluate its performance after unlearning specific concepts.

### 1. Change Directory

Navigate to the evaluation folder:

```bash
cd $REPO_ROOT/Evaluation/UnlearnCanvas
conda activate cuig
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
```bash
python sample.py \
    --unet_ckpt_path "/fs/scratch/PAS2099/lee.10369/CUIG/mace/models/continual/grad_proj/100/style/steps50_lr1e5/thruByzantine/LoRA_fusion_model/unet/diffusion_pytorch_model.safetensors" \
    --output_dir "$OUTPUT_ROOT/mace/eval_results/continual/grad_proj/100/style/steps50_lr1e5/thruByzantine_1e1/images" \
    --styles_subset '["Byzantine", "Blossom_Season", "Rust", "Crayon", "Fauvism", "Superstring", "Red_Blue_Ink", "Gorgeous_Love", "French", "Joy", "Greenfield", "Expressionism", "Impressionism"]' \
    --objects_subset '["Architectures", "Butterfly", "Flame", "Flowers", "Horses", "Human", "Sea", "Trees"]' \
    --pipeline_dir $BASE_MODEL_DIR \
    --seed 188
```
```bash
python sample.py \
    --unet_ckpt_path $OUTPUT_ROOT/fmn/models/independent/base/object/Bears/delta.bin \
    --output_dir $OUTPUT_ROOT/fmn/eval_results/independent/base/object/Bears/images \
    --styles_subset '["Blossom_Season", "Rust", "Crayon", "Fauvism", "Superstring", "Red_Blue_Ink", "Gorgeous_Love", "French", "Joy", "Greenfield", "Expressionism", "Impressionism"]' \
    --objects_subset '["Bears"]' \
    --pipeline_dir $BASE_MODEL_DIR 
```

### 3. Evaluate Accuracy

Run the evaluator on the sampled images:

```bash
python evaluate.py \
    --input_dir /fs/scratch/PAS2099/lee.10369/CUIG/ca/eval_results/continual/l1sp/style/50_base/thruCartoon/images \
    --output_dir /fs/scratch/PAS2099/lee.10369/CUIG/ca/eval_results/continual/l1sp/style/50_base/thruCartoon/metrics \
    --classifier_dir $REPO_ROOT/Evaluation/UnlearnCanvas/classifiers \
    --unlearn '["Abstractionism", "Byzantine", "Cartoon"]' \
    --retain '["Blossom_Season", "Rust", "Crayon", "Fauvism", "Superstring", "Red_Blue_Ink", "Gorgeous_Love", "French", "Joy", "Greenfield", "Expressionism", "Impressionism"]' \
    --cross_retain '["Architectures", "Butterfly", "Flame", "Flowers", "Horses", "Human", "Sea", "Trees"]'
```
```bash
python evaluate.py \
    --input_dir $OUTPUT_ROOT/fmn/eval_results/independent/base/object/Bears/images \
    --output_dir $OUTPUT_ROOT/fmn/eval_results/independent/base/object/Bears/metrics \
    --classifier_dir $REPO_ROOT/Evaluation/UnlearnCanvas/classifiers \
    --unlearn '["Bears"]' \
    --cross_retain '["Blossom_Season", "Rust", "Crayon", "Fauvism", "Superstring", "Red_Blue_Ink", "Gorgeous_Love", "French", "Joy", "Greenfield", "Expressionism", "Impressionism"]' \
    --retain '["Architectures", "Butterfly", "Flame", "Flowers", "Horses", "Human", "Sea", "Trees"]'
```
```bash
python evaluate.py \
    --input_dir $OUTPUT_ROOT/mace/eval_results/continual/grad_proj/100/style/steps50_lr1e5/thruByzantine_1e1/images \
    --output_dir $OUTPUT_ROOT/mace/eval_results/continual/grad_proj/100/style/steps50_lr1e5/thruByzantine_1e1/metrics \
    --classifier_dir $REPO_ROOT/Evaluation/UnlearnCanvas/classifiers \
    --unlearn '["Byzantine"]' \
    --cross_retain '["Architectures", "Butterfly", "Flame", "Flowers", "Horses", "Human", "Sea", "Trees"]' \
    --retain '["Blossom_Season", "Rust", "Crayon", "Fauvism", "Superstring", "Red_Blue_Ink", "Gorgeous_Love", "French", "Joy", "Greenfield", "Expressionism", "Impressionism"]' \
    --seed 188
```

**Notes:**
- Adjust the paths and style/object lists as needed for your experiments.
- Ensure all required dependencies are installed before running the scripts.
