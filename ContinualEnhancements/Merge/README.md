# Model Merging

This section explains how to use various model **merging strategies** in this repository. These tools allow you to combine two or more diffusion models into a single model using methods like Uniform Averaging, Task Arithmetic and TIES.

---

```bash
cd $REPO_ROOT/ContinualEnhancements/Merge
```

## TIES
```bash
python main.py \
--base_model_dir $REPO_ROOT/UnlearningMethods/base_models/UnlearnCanvas \
--ckpt_paths '["/fs/scratch/PAS2099/lee.10369/CUIG/esd/models/independent/base/style/CompvisToDiffusers/Abstractionism.pth",
                "/fs/scratch/PAS2099/lee.10369/CUIG/esd/models/independent/base/style/CompvisToDiffusers/Byzantine.pth",
                "/fs/scratch/PAS2099/lee.10369/CUIG/esd/models/independent/base/style/CompvisToDiffusers/Cartoon.pth",
                "/fs/scratch/PAS2099/lee.10369/CUIG/esd/models/independent/base/style/CompvisToDiffusers/Cold_Warm.pth",
                "/fs/scratch/PAS2099/lee.10369/CUIG/esd/models/independent/base/style/CompvisToDiffusers/Ukiyoe.pth",
                "/fs/scratch/PAS2099/lee.10369/CUIG/esd/models/independent/base/style/CompvisToDiffusers/Van_Gogh.pth",
                "/fs/scratch/PAS2099/lee.10369/CUIG/esd/models/independent/base/style/CompvisToDiffusers/Neon_Lines.pth",
                "/fs/scratch/PAS2099/lee.10369/CUIG/esd/models/independent/base/style/CompvisToDiffusers/Picasso.pth",
                "/fs/scratch/PAS2099/lee.10369/CUIG/esd/models/independent/base/style/CompvisToDiffusers/On_Fire.pth",
                "/fs/scratch/PAS2099/lee.10369/CUIG/esd/models/independent/base/style/CompvisToDiffusers/Magic_Cube.pth",
                "/fs/scratch/PAS2099/lee.10369/CUIG/esd/models/independent/base/style/CompvisToDiffusers/Winter.pth",
                "/fs/scratch/PAS2099/lee.10369/CUIG/esd/models/independent/base/style/CompvisToDiffusers/Vibrant_Flow.pth"]' \
--save_path "/fs/scratch/PAS2099/lee.10369/CUIG/esd/models/independent/merge/ties/style/lambda1.70_topk0.80/thruVibrantFlow.pth" \
--merge_method 'ties' \
--device 'cpu' \
--key_filter 'attn2' \
--ties_lambda 1.70 \
--ties_topk 0.80 \
--ties_merge_func 'mean'
```


---

