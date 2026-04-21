## 🔬 Evaluation

Follow these steps to sample from a model and evaluate its performance after unlearning a celebrity

### 1. Change Directory

Navigate to the evaluation folder:

```bash
cd $REPO_ROOT/Evaluation/Celebrity
conda activate cuig
```

### 2. Sample Images

Example: sampling from a model that has already sequentially unlearned the celebrities **Neil Degrasse Tyson**, **Benicio Del Toro**, and **Aziz Ansari**:

Generate prompts
```bash
python generate_celeb_prompts.py \
    --prompt "Natalie Portman" \
    --output_path "prompts/Natalie_Portman.txt"
```

Generate image outputs of celebrity (can use either SDXL or standard SD)
```bash
python sample_celeb.py \
    --model_family "sdxl" \
    --ckpt "SD" \
    --output_dir "$OUTPUT_ROOT/Independent/Celebrity/stablediffusion/v1.4/images" \
    --celeb_subset '["Neil_Degrasse_Tyson", "Benicio_Del_Toro", "Aziz_Ansari", "Oprah_Winfrey", "Betty_White", "Megan_Fox", "Morgan_Freeman", "Keanu_Reeves", "George_Takei", "Aretha_Franklin", "Maya_Angelou", "Natalie_Portman"]'
```
    
    
### 3. Evaluate Accuracy

### Setup

1. Navigate to the evaluation directory:
    ```bash
    cd $REPO_ROOT/Evaluation/Celebrity
    ```

2. Download and extract resources:
    ```bash
    curl -L https://s3.amazonaws.com/giphy-public/models/celeb-detection/resources.tar.gz -o resources.tar.gz
    tar -xzf resources.tar.gz
    rm resources.tar.gz
    ```

3. Configure environment variables in `$REPO_ROOT/Evaluation/Celebrity/celeb-detection-oss/.env`:
    - Set `$APP_DATA_DIR` to the extracted `resources` folder
    - Set `$APP_RECOGNITION_WEIGHTS_FILE` to `resources/face_recognition/best_model_states.pkl`

4. Create and activate the conda environment:
    ```bash
    conda create -n celeb python=3.6.13
    pip install -r $REPO_ROOT/Evaluation/Celebrity/celeb-detection-oss/requirements_gpu.txt
    ```


```bash
conda activate celeb
cd $REPO_ROOT/Evaluation/Celebrity/celeb-detection-oss
```

```bash
python examples/evaluate_celeb.py \
    --input_dir "/fs/scratch/PAS2099/lee.10369/CUIG/stablediffusion/v1.4/celebrity/images" \
    --output_dir "/fs/scratch/PAS2099/lee.10369/CUIG/stablediffusion/v1.4/celebrity/metrics" \
    --unlearn '[]' \
    --retain '["Morgan_Freeman", "Keanu_Reeves", "George_Takei", "Aretha_Franklin", "Maya_Angelou", "Natalie_Portman"]'
```

```bash
python examples/evaluate_celeb.py \
    --input_dir "/fs/scratch/PAS2099/lee.10369/CUIG/stablediffusion/sdxl/celebrity/images" \
    --output_dir "/fs/scratch/PAS2099/lee.10369/CUIG/stablediffusion/sdxl/celebrity/metrics" \
    --unlearn '[]' \
    --retain '["Neil_Degrasse_Tyson", "Benicio_Del_Toro", "Aziz_Ansari", "Oprah_Winfrey", "Betty_White", "Megan_Fox", "Morgan_Freeman", "Keanu_Reeves", "George_Takei", "Aretha_Franklin", "Maya_Angelou", "Natalie_Portman"]'
```

Evaluate MS-COCO Score

```bash
conda activate cuig
cd $REPO_ROOT/Evaluation/Celebrity
```

```bash
python coco/evaluate_coco.py \
    --input_dir "/fs/scratch/PAS2099/lee.10369/CUIG/stablediffusion/v2.1/ms_coco/images" \
    --output_dir "/fs/scratch/PAS2099/lee.10369/CUIG/stablediffusion/v2.1/ms_coco/metrics"
```