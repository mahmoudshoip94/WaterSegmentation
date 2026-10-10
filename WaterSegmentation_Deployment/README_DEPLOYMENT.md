# Water Segmentation — Deployment

Flask application for predicting water pixels from a **12-band, 128 × 128 TIFF/GeoTIFF** using the project's pretrained **U-Net + ResNet34** baseline.

## What is included

```text
WaterSegmentation_Deployment/
├── app.py                     # Flask web app and POST APIs
├── inference.py               # model loading, exact training normalization, prediction
├── requirements.txt
├── model/
│   └── README.md              # where to place the checkpoint
├── templates/
│   └── index.html             # upload UI, RGB reference, prediction mask
├── scripts/
│   ├── predict_one.py         # predict one TIFF from the command line
│   └── test_api.py            # test the running API on >=3 real TIFFs
└── outputs/                   # generated test masks (not committed)
```

## 1. Put the checkpoint in place

Copy the trained `best_Unet.pt` into:

```text
model/best_Unet.pt
```

The checkpoint is intentionally **not included** in this source ZIP. It is approximately 93.5 MB and was not attached to this working session. Alternatively, set `MODEL_PATH` to the checkpoint's actual path.

Use the checkpoint for the **12-channel pretrained baseline**. Do not use the weights for the 12→3 CNN Converter experiment or the 13-channel experiment.

## 2. Create an environment and install packages

Recommended: Python 3.11. Use a clean virtual environment. Install a matching PyTorch/torchvision build for your OS and CPU/GPU using the official PyTorch installation selector, then install the remaining dependencies.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -r requirements.txt
```

If PyTorch/torchvision are already installed in this virtual environment, confirm that their versions are compatible before reinstalling them. The project can run on CPU; CUDA is optional.

## 3. Start the web app

Run from this directory (the directory containing `app.py`):

```powershell
python app.py
```

Open `http://127.0.0.1:5000` in your browser. The first prediction loads the checkpoint and may take longer than later predictions.

To use a checkpoint at another location:

```powershell
$env:MODEL_PATH = "D:\path\to\best_Unet.pt"
python app.py
```

Optional environment variables:

- `MODEL_PATH`: checkpoint path; defaults to `model/best_Unet.pt`.
- `DEVICE`: explicit PyTorch device such as `cpu` or `cuda`; otherwise auto-detected.
- `PORT`: local Flask port; defaults to `5000`.
- `MAX_UPLOAD_MB`: upload size limit; defaults to `50`.

## 4. API endpoints

### `GET /health`

Reports whether the checkpoint exists and whether the model has been loaded.

### `POST /preview`

Multipart form field: `file`. Returns an RGB PNG preview made from bands Red (index 3), Green (index 2), and Blue (index 1). Per-image percentile stretching is used only for display, never for model input.

### `POST /predict`

Multipart form field: `file`. Accepts `.tif` or `.tiff` and returns an `image/png` binary mask. The PNG has pixel value `255` for predicted water and `0` for background. Useful headers include `X-Water-Pixel-Percent`, `X-Water-Pixel-Count`, `X-Total-Pixel-Count`, and `X-Model-Threshold`.

Example using cURL:

```bash
curl -X POST -F "file=@data/images/example.tif" \
  http://127.0.0.1:5000/predict --output water_mask.png
```

Example using Python `requests`:

```python
import requests

with open("data/images/example.tif", "rb") as f:
    response = requests.post(
        "http://127.0.0.1:5000/predict",
        files={"file": ("example.tif", f, "image/tiff")},
        timeout=180,
    )
response.raise_for_status()
with open("water_mask.png", "wb") as f:
    f.write(response.content)
print("Water coverage (%):", response.headers.get("X-Water-Pixel-Percent"))
```

## 5. Inference behavior (must match training)

- Architecture: `segmentation_models_pytorch.Unet`, `encoder_name="resnet34"`, `in_channels=12`, `classes=1`, `activation=None`.
- The loaded state_dict includes the adapted 12-channel first convolution; `encoder_weights=None` is used at reconstruction time because trained weights are loaded immediately afterward.
- Expected input shape: `(12, 128, 128)`.
- Normalized channels: `[0, 1, 2, 3, 4, 5, 6, 8, 9, 11]`.
- QA channel 7 and ESA WorldCover channel 10 stay unnormalized.
- Dataset-level training means and standard deviations are hard-coded in `inference.py` exactly as supplied from the notebook.
- Inference does not apply augmentation.
- The model returns logits; inference applies sigmoid then `probability > 0.5`.
- Sentinel values such as `-9999` are not specially replaced in the deployment path because the supplied Dataset `__getitem__` did not contain that operation. If the actual training pipeline applies an upstream nodata transformation before the Dataset, replicate that transformation here before claiming preprocessing parity.

## 6. Single-image inference

```powershell
python scripts/predict_one.py ..\data\images\example.tif --output outputs\example_mask.png
```

You may provide an explicit checkpoint with `--model`.

## 7. Test at least three images through the running API

First start `python app.py` in one terminal. In another terminal, from this deployment directory:

```powershell
python scripts/test_api.py --images-dir ..\data\images --count 3  # when these files live in a deployment/ subfolder
# If you copied the deployment files to the repository root instead, use:
# python scripts/test_api.py --images-dir data\images --count 3
```

The test script submits three real TIFF files to `/predict`, checks HTTP status/content type, input band count, output dimensions, binary values, and saves masks under `outputs/api_tests/`. Run it against the actual dataset before reporting that these tests passed; the source ZIP does not contain the private dataset or checkpoint.

You can also check the health endpoint:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/health
```

## 8. Reported experiment results

The project README records the 12-channel pretrained baseline (70/15/15 split) at validation IoU `0.6884`, test IoU `0.7071`, and test F1 `0.8284`, using threshold `0.5`. The experiment comparison shared during preparation recorded loss `0.1962`, accuracy `0.9288`, precision `0.8827`, recall `0.7804`, F1 `0.8284`, and IoU `0.7071` for this baseline; keep these scores labeled with the evaluation split/protocol used by the notebook.

The CNN Converter alternative scored IoU `0.6376` and F1 `0.7787` in the shared comparison, so the 12-channel baseline is selected for deployment. Do not claim the web interface re-evaluates test-set metrics: it returns predictions for submitted images.

## 9. GitHub

From the existing repository root, copy these source files/folders into the repository (you can keep them at the root or inside a dedicated `deployment/` directory, but keep `templates/` relative to `app.py`). Then review the changes:

```bash
git status
git add app.py inference.py requirements.txt README_DEPLOYMENT.md templates scripts model/README.md .gitignore outputs/.gitkeep
git commit -m "Add Flask deployment for multispectral water segmentation"
git push origin main
```

`model/*.pt` is ignored intentionally because the checkpoint is large. For hosting elsewhere, make sure the runtime can access the checkpoint via mounted storage, a private model artifact, or Git LFS. Never claim the deployed app is live until you have actually hosted it and verified its public URL.

## Limitations

- The checkpoint and the 306-image dataset are not bundled.
- Inputs are intentionally restricted to 12-band 128 × 128 rasters because that is the model's training input shape.
- The RGB preview is only a visual reference; it is not the 3-channel CNN Converter and is not passed to the model.
- `python app.py` is suitable for local testing, not as a production server exposed directly to the public internet. Use a production WSGI server and appropriate upload/authentication limits for public hosting.
