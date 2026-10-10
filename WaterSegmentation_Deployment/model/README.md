# Model checkpoint

Copy the trained `best_Unet.pt` checkpoint into this directory:

```text
model/best_Unet.pt
```

The checkpoint is not included in the deployment source ZIP because it is approximately 93.5 MB and was not attached to this working session. The application will report `model_missing` until the file is copied here, or until the `MODEL_PATH` environment variable points to it.

The checkpoint must be the **12-channel pretrained U-Net + ResNet34 baseline**, not the CNN Converter model or the 13-channel model.
