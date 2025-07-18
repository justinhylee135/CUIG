# Model Merging

This section explains how to use various model **merging strategies** in this repository. These tools allow you to combine two or more diffusion models into a single model using methods like Uniform Averaging, Task Arithmetic and TIES.

---

```bash
cd $REPO_ROOT/ContinualEnhancements/Merge
```


# Example Merge
## TIES Merging for CA Object Checkpoints
```bash
python main.py \
--base_model_dir $REPO_ROOT/UnlearningMethods/base_models/UnlearnCanvas \
--ckpt_paths '["$OUTPUT_ROOT/ca/models/independent/base/object/Bears/delta.bin",
                "$OUTPUT_ROOT/ca/models/independent/base/object/Birds/delta.bin",
                "$OUTPUT_ROOT/ca/models/independent/base/object/Cats/delta.bin",
                "$OUTPUT_ROOT/ca/models/independent/base/object/Dogs/delta.bin"]' \
--save_path "$OUTPUT_ROOT/ca/models/independent/merge/ties/object/lambda1.25_topk0.80/thruDogs.pth" \
--merge_method 'ties' \
--key_filter '["attn2.to_k", "attn2.to_v"]'
--device 'cpu' \
--ties_lambda 1.25 \
--ties_topk 0.80 \
--ties_merge_func 'mean'
```

---

