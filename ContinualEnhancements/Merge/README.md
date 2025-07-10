# Model Merging

This section explains how to use various model **merging strategies** in this repository. These tools allow you to combine two or more diffusion models into a single model using methods like Uniform Averaging, Task Arithmetic and TIES.

---

```bash
cd $REPO_ROOT/ContinualEnhancements/Merge
```

## TIES
### ESD
#### Style
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
#### Object
```bash
python main.py \
--base_model_dir $REPO_ROOT/UnlearningMethods/base_models/UnlearnCanvas \
--ckpt_paths '["/fs/scratch/PAS2099/lee.10369/CUIG/esd/models/independent/base/object/CompvisToDiffusers/lr_5e6/ddim_linear_ng_4_sg_7.5/Bears.pth",
                "/fs/scratch/PAS2099/lee.10369/CUIG/esd/models/independent/base/object/CompvisToDiffusers/lr_5e6/ddim_linear_ng_4_sg_7.5/Birds.pth",
                "/fs/scratch/PAS2099/lee.10369/CUIG/esd/models/independent/base/object/CompvisToDiffusers/lr_5e6/ddim_linear_ng_4_sg_7.5/Cats.pth",
                "/fs/scratch/PAS2099/lee.10369/CUIG/esd/models/independent/base/object/CompvisToDiffusers/lr_5e6/ddim_linear_ng_4_sg_7.5/Dogs.pth",
                "/fs/scratch/PAS2099/lee.10369/CUIG/esd/models/independent/base/object/CompvisToDiffusers/lr_5e6/ddim_linear_ng_4_sg_7.5/Fishes.pth",
                "/fs/scratch/PAS2099/lee.10369/CUIG/esd/models/independent/base/object/CompvisToDiffusers/lr_5e6/ddim_linear_ng_4_sg_7.5/Frogs.pth"]' \
--save_path "/fs/scratch/PAS2099/lee.10369/CUIG/esd/models/independent/merge/ties/object/lambda1.00_topk0.30/thruFrogs.pth" \
--merge_method 'ties' \
--device 'cpu' \
--ties_lambda 1.00 \
--ties_topk 0.30 \
--ties_merge_func 'mean'
```

### CA
#### Style
#### Object
```bash
python main.py \
--base_model_dir $REPO_ROOT/UnlearningMethods/base_models/UnlearnCanvas \
--ckpt_paths '["/fs/scratch/PAS2099/lee.10369/CUIG/ca/models/independent/base/object/no_retention_bsz_8/Bears/delta.bin",
                "/fs/scratch/PAS2099/lee.10369/CUIG/ca/models/independent/base/object/no_retention_bsz_8/Birds/delta.bin",
                "/fs/scratch/PAS2099/lee.10369/CUIG/ca/models/independent/base/object/no_retention_bsz_8/Cats/delta.bin",
                "/fs/scratch/PAS2099/lee.10369/CUIG/ca/models/independent/base/object/no_retention_bsz_8/Dogs/delta.bin"]' \
--save_path "/fs/scratch/PAS2099/lee.10369/CUIG/ca/models/independent/merge/ties/object/lambda1.25_topk0.80/thruDogs.pth" \
--merge_method 'ties' \
--key_filter '["attn2.to_k", "attn2.to_v"]'
--device 'cpu' \
--ties_lambda 1.25 \
--ties_topk 0.80 \
--ties_merge_func 'mean'
```

---

