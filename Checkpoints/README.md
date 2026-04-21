# Set Up

## Classifier
Add all classifier models for evaluation here.
The instructions below show how to download the model used in our paper. This model is from the UnlearnCanvas benchmark.

```bash
cd $REPO_ROOT/Checkpoints
mkdir Classifiers
cd Classifiers
gdown --folder https://drive.google.com/drive/folders/1AoazlvDgWgc3bAyHDpqlafqltmn4vm61
mv cls_model UnlearnCanvas
cd UnlearnCanvas
rm style60.pth
mv style50.pth style_classifier.pth
mv style50_cls.pth object_classifier.pth
```


## Generator
Add all image generation models here.
The instructions below show how to download the model used in our paper. This model is from the UnlearnCanvas benchmark.

```bash
cd $REPO_ROOT/Checkpoints
mkdir Generators
cd Generators
gdown --folder https://drive.google.com/drive/folders/18x40pLBcfNFyxBWZBGncTjqJTs_75SLx
mv style50 UnlearnCanvas
```

After this you can set your $BASE_MODEL_DIR to $REPO_ROOT/Checkpoints/Generators/UnlearnCanvas