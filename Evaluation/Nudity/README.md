## 🔬 Evaluation

Follow these steps to sample from a model and evaluate its performance after unlearning nudity

### 1. Change Directory

Navigate to the evaluation folder:

```bash
cd $REPO_ROOT/Evaluation/Nudity
conda activate cuig
```

### 2. Sample Images

Example: sampling from a model that has sequentially unlearned the keywords **Nudity**, **Naked**, and **Erotic**:

```bash
python sample_from_csv.py \
    --model_name "SD" \
    --output_dir "/fs/scratch/PAS2099/lee.10369/CUIG/stablediffusion/v1.4/images" \
    --num_prompts 200 \
    --prompts_path "nudity_benchmark.csv"
```

### 3. Evaluate Accuracy

Run the nudenet classifier on the sampled images:

```bash
python evaluate_nudenet.py \
    --input_dir "/fs/scratch/PAS2099/lee.10369/CUIG/stablediffusion/v1.4/images" \
    --output_dir "/fs/scratch/PAS2099/lee.10369/CUIG/stablediffusion/v1.4/metrics"
```

### 4. FID and CLIP Score on MS_COCO_30k
COCO_CAPTIONS_URL = 'https://github.com/boomb0om/text2image-benchmark/releases/download/v0.0.1/MS-COCO_val2014_30k_captions.csv'
COCO_FID_STATS_URL = 'https://github.com/boomb0om/text2image-benchmark/releases/download/v0.0.1/MS-COCO_val2014_fid_stats.npz'

```bash
python sample_from_csv.py \
    --model_name "SD" \
    --output_dir "/fs/scratch/PAS2099/lee.10369/CUIG/stablediffusion/v1.4/coco_images" \
    --num_prompts 200 \
    --prompts_path "ms_coco.csv"
```

```bash
python evaluate_coco.py \
    --input_dir "/fs/scratch/PAS2099/lee.10369/CUIG/stablediffusion/v1.4/coco_images" \
    --output_dir "/fs/scratch/PAS2099/lee.10369/CUIG/stablediffusion/v1.4/coco_metrics"
```


**Notes:**
- Adjust the paths and style/object lists as needed for your experiments.
- Ensure all required dependencies are installed before running the scripts.
