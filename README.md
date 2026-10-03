# 🌊 Water Segmentation Using Multispectral and Optical Data

A deep-learning semantic segmentation project for detecting water bodies from **12-channel satellite raster data** using a **U-Net implemented from scratch in PyTorch**.

The project combines multispectral, optical, terrain, land-cover, and water-occurrence information, with a classical **NDWI threshold baseline** used for comparison.

<p align="center">
  <img src="figures/fig1_rgb_gt.png" alt="RGB Composite and Ground Truth" width="850">
</p>

---

## 🎯 Project Overview

The goal of this project is to perform **binary semantic segmentation of water pixels** from multispectral satellite imagery.

Each input sample is a `128 × 128` raster with **12 channels**, while the target is a binary mask:

* `0` → Background
* `1` → Water

The project follows an experimental workflow:

```text
Dataset Exploration
        ↓
Preprocessing
        ↓
Classical NDWI Baseline
        ↓
Baseline U-Net
        ↓
Feature Engineering
        ↓
Channel Ablation & Selection
        ↓
Final Model Experiments
        ↓
Independent Evaluation
```

---

## 🛰️ Dataset

The dataset contains **306 image-mask pairs**.

| Property                |       Value |
| ----------------------- | ----------: |
| Raster Images           |         306 |
| Binary Masks            |         306 |
| Image Size              | `128 × 128` |
| Input Channels          |          12 |
| Images Containing Water |         261 |
| Images Without Water    |          45 |
| Water Pixels            |   1,302,272 |
| Background Pixels       |   3,711,232 |
| Water Coverage          |      25.98% |
| Background Coverage     |      74.02% |

### Channel Configuration

| Index | Channel                      |
| ----: | ---------------------------- |
|     0 | Coastal Aerosol              |
|     1 | Blue                         |
|     2 | Green                        |
|     3 | Red                          |
|     4 | NIR                          |
|     5 | SWIR1                        |
|     6 | SWIR2                        |
|     7 | QA Band                      |
|     8 | MERIT DEM                    |
|     9 | Copernicus DEM               |
|    10 | ESA WorldCover               |
|    11 | Water Occurrence Probability |

### RGB Visualization

For visualization, the RGB composite uses:

```text
Red   → Channel 3
Green → Channel 2
Blue  → Channel 1
```

### Data Handling

Two channels are treated as categorical features and are not normalized:

* QA Band
* ESA WorldCover

The remaining channels are treated as continuous features.

Invalid MERIT DEM values of `-9999` are handled during preprocessing, while valid negative spectral values are preserved.

---

## 🔬 Exploratory Data Analysis

The EDA stage was used to understand:

* Channel distributions and ranges
* RGB composites
* Ground-truth masks
* Per-image statistics
* Channel correlations
* Categorical and continuous feature behavior
* Invalid DEM values

<p align="center">
  <img src="figures/eda_rgb_composite.png" alt="RGB Composite" width="700">
</p>

<p align="center">
  <img src="figures/eda_12_bands.png" alt="12 Satellite Channels" width="850">
</p>

---

## 🧪 Classical Baseline — NDWI

Before training the deep-learning model, a classical **NDWI thresholding approach** was implemented as a reference baseline.

NDWI was calculated as:

```python
NDWI = (Green - NIR) / (Green + NIR + 1e-6)
```

The threshold was selected using the validation set.

### NDWI Results

| Metric | Validation |   Test |
| ------ | ---------: | -----: |
| IoU    |     0.7067 | 0.5052 |
| F1     |     0.8282 | 0.6712 |

The best validation threshold was:

```text
Threshold = -0.3
```

<p align="center">
  <img src="figures/fig3_ndwi_baseline.png" alt="NDWI Baseline" width="850">
</p>

This baseline provides a useful reference for evaluating whether the deep-learning pipeline adds value beyond a simple spectral index rule.

---

## 🧠 Deep Learning Approach

The main segmentation model is a **U-Net implemented from scratch in PyTorch**.

The project contains multiple experimental stages rather than a single training run.

### Baseline U-Net

The baseline model uses:

* 12 raw input channels
* BCEWithLogitsLoss
* Adam optimizer
* Learning rate `1e-4`
* Batch size `8`
* Early stopping
* Horizontal and vertical flips
* Rot90 augmentation
* Light numerical-channel noise

### Baseline Results

| Metric    | Validation |   Test |
| --------- | ---------: | -----: |
| IoU       |     0.6176 | 0.6475 |
| F1        |          — | 0.7861 |
| Accuracy  |          — | 0.9117 |
| Precision |          — | 0.8428 |
| Recall    |          — | 0.7365 |

---

## 🧩 Feature Engineering & Channel Selection

Feature engineering was used to investigate whether spectral indices and channel selection could improve segmentation.

The experiments included:

* NDWI
* MNDWI
* NDMI
* Channel ablation
* Greedy channel removal

### Channel Selection

The selected configuration removed:

```text
Coastal Aerosol
ESA WorldCover
```

and added:

```text
NDWI
MNDWI
```

resulting in a final **12-channel input configuration**.

### Feature Engineering Results

| Metric    | Validation |   Test |
| --------- | ---------: | -----: |
| IoU       |     0.6453 | 0.6505 |
| F1        |     0.7844 | 0.7882 |
| Precision |     0.8075 | 0.8498 |
| Recall    |     0.7626 | 0.7350 |
| Accuracy  |          — | 0.9130 |

The best validation checkpoint was reached at **epoch 40**.

---

## 🚀 Final Model Experiments

Two final-model experiments were conducted using:

```text
12 raw channels + NDWI
```

with:

* U-Net from scratch
* Base filters = `32`
* Depth = `5`
* Dropout = `0.1`
* BCE + Dice loss
* Adam optimizer
* Learning rate = `1e-4`

### Experiment 1 — Independent Test Set

Split:

```text
80% Train / 10% Validation / 10% Test
```

The best validation threshold was:

```text
0.6
```

#### Results

| Metric    | Validation | Independent Test |
| --------- | ---------: | ---------------: |
| IoU       |     0.7479 |           0.5206 |
| F1        |     0.8558 |           0.6848 |
| Precision |     0.9423 |                — |
| Recall    |     0.7838 |                — |
| Accuracy  |     0.9387 |                — |

This experiment includes an independent test set, and the reported test metrics were obtained from that held-out split.

---

### Experiment 2 — 80/20 Validation Setup

Split:

```text
80% Train / 20% Validation
```

This experiment does **not** contain an independent test set.

The best validation threshold was again:

```text
0.6
```

#### Results

| Metric    | Validation |
| --------- | ---------: |
| IoU       |     0.6991 |
| F1        |     0.8229 |
| Precision |     0.8840 |
| Recall    |     0.7697 |
| Accuracy  |     0.9166 |

These values are validation results after threshold tuning and should not be presented as independent test performance.

---

## 📊 Experiment Summary

| Experiment          | Split    | Input                       | Validation IoU | Test IoU |
| ------------------- | -------- | --------------------------- | -------------: | -------: |
| NDWI Baseline       | 70/15/15 | NDWI                        |         0.7067 |   0.5052 |
| Baseline U-Net      | 70/15/15 | 12 raw                      |         0.6176 |   0.6475 |
| Feature Engineering | 70/15/15 | Selected raw + NDWI + MNDWI |         0.6453 |   0.6505 |
| Final Experiment 1  | 80/10/10 | 12 raw + NDWI               |         0.7479 |   0.5206 |
| Final Experiment 2  | 80/20    | 12 raw + NDWI               |         0.6991 |        — |

> **Note:** These experiments use different data splits and evaluation protocols. Validation and test metrics from different experiments should therefore not be treated as a single directly comparable leaderboard.

---

## 🖼️ Visual Results

### RGB + Ground Truth

<p align="center">
  <img src="figures/fig1_rgb_gt.png" alt="RGB and Ground Truth" width="850">
</p>

### Multispectral Information

<p align="center">
  <img src="figures/fig2_multispectral_bands.png" alt="Multispectral Bands" width="850">
</p>

### U-Net Prediction

<p align="center">
  <img src="figures/fig4_unet_prediction.png" alt="U-Net Prediction" width="850">
</p>

### Training Curves

<p align="center">
  <img src="figures/fig5_training_curves.png" alt="Training Curves" width="850">
</p>

### NDWI vs U-Net

<p align="center">
  <img src="figures/fig6_ndwi_vs_unet.png" alt="NDWI vs U-Net" width="850">
</p>

### Error Visualization

<p align="center">
  <img src="figures/fig7_error_map.png" alt="Prediction Error Map" width="850">
</p>

---

## 🧠 Key Findings

### 1. Spectral indices were useful but not sufficient on their own

The classical NDWI baseline achieved strong validation performance, but its test performance was lower, showing the difference between fitting a validation threshold and generalizing to unseen scenes.

### 2. Channel selection changed the input representation

Feature engineering experiments showed that removing some channels and adding spectral indices could produce a different and competitive representation.

### 3. Evaluation protocol matters

The project includes experiments with different train/validation/test splits. Because of this, reported metrics must always be interpreted together with their evaluation setup.

### 4. Independent testing reveals a clear generalization gap

The first final-model experiment achieved stronger validation metrics than its independent test metrics, highlighting the importance of evaluating segmentation models on data that was not used for model or threshold selection.

---

## 📁 Project Structure

```text
WaterSegmentation/
│
├── data/
│   ├── images/
│   └── labels/
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── prepro&model.ipynb
│   ├── prepro&model_4.ipynb
│   ├── final_model.ipynb
│   └── final_model_2.ipynb
│
├── figures/
│   ├── eda_rgb_composite.png
│   ├── eda_label_mask.png
│   ├── eda_12_bands.png
│   ├── eda_channel_minmax.png
│   ├── eda_correlation_matrix.png
│   ├── fig1_rgb_gt.png
│   ├── fig2_multispectral_bands.png
│   ├── fig3_ndwi_baseline.png
│   ├── fig4_unet_prediction.png
│   ├── fig5_training_curves.png
│   ├── fig6_ndwi_vs_unet.png
│   └── fig7_error_map.png
│
└── README.md
```

---

## 🔧 Technical Details

<details>
<summary><strong>Preprocessing & Data Splits</strong></summary>

### Preprocessing

The preprocessing pipeline includes:

* Raster loading with `rasterio`
* Binary mask loading with Pillow
* Invalid MERIT DEM handling
* Dataset-wise per-channel normalization using training data only
* Preservation of categorical channels
* Optional spectral indices
* Training-only augmentation
* `SEED = 42`

### Baseline Split

```text
Train      = 214
Validation = 46
Test       = 46
```

### Final Experiment 1 Split

```text
Train      = 244
Validation = 31
Test       = 31
```

### Final Experiment 2 Split

```text
Train      = 244
Validation = 62
Test       = None
```

</details>

<details>
<summary><strong>Feature Engineering Details</strong></summary>

### NDWI

```python
NDWI = (Green - NIR) / (Green + NIR + 1e-6)
```

### Additional Features Explored

```text
NDWI
MNDWI
NDMI
```

### Ablation & Greedy Search

The experiments evaluated the impact of removing channels and adding spectral indices.

The selected configuration was:

```text
Raw:
[1,2,3,4,5,6,7,8,9,11]

Added:
NDWI
MNDWI

Removed:
Coastal Aerosol
ESA WorldCover
```

Final input size:

```text
12 channels
```

</details>

<details>
<summary><strong>Baseline U-Net Configuration</strong></summary>

| Parameter               | Value             |
| ----------------------- | ----------------- |
| Input Channels          | 12                |
| Output Channels         | 1                 |
| Base Filters            | 8                 |
| Depth                   | 6                 |
| Parameters              | 7,783,073         |
| Loss                    | BCEWithLogitsLoss |
| Optimizer               | Adam              |
| Learning Rate           | `1e-4`            |
| Batch Size              | 8                 |
| Maximum Epochs          | 60                |
| Early Stopping Patience | 10                |

Augmentation:

```text
Horizontal Flip
Vertical Flip
Rot90
Numerical Channel Noise = 0.05
```

</details>

<details>
<summary><strong>Final Model Configuration</strong></summary>

### Shared Architecture

```text
Input Channels = 13
Base Filters   = 32
Depth          = 5
Dropout        = 0.1
Output         = 1
Loss           = BCE + Dice
```

### Final Experiment 1

| Parameter         | Value             |
| ----------------- | ----------------- |
| Split             | 80/10/10          |
| Optimizer         | Adam              |
| Learning Rate     | `1e-4`            |
| Weight Decay      | `1e-4`            |
| Batch Size        | 8                 |
| `pos_weight`      | ≈ 2.8228          |
| Scheduler         | CosineAnnealingLR |
| Gradient Clipping | 1.0               |
| Early Stopping    | 20                |
| Maximum Epochs    | 100               |
| Selection Metric  | Validation IoU    |
| Best Epoch        | 14                |
| Early Stopping    | Epoch 34          |

### Final Experiment 2

| Parameter          | Value             |
| ------------------ | ----------------- |
| Split              | 80/20             |
| Optimizer          | Adam              |
| Learning Rate      | `1e-4`            |
| Weight Decay       | `1e-4`            |
| Batch Size         | 8                 |
| `pos_weight`       | ≈ 2.8200          |
| Scheduler          | ReduceLROnPlateau |
| Scheduler Factor   | 0.5               |
| Scheduler Patience | 5                 |
| Minimum LR         | `1e-7`            |
| Gradient Clipping  | 1.0               |
| Early Stopping     | 20                |
| Maximum Epochs     | 100               |
| Selection Metric   | Validation IoU    |
| Best Epoch         | 47                |
| Early Stopping     | Epoch 67          |

</details>

<details>
<summary><strong>Notebook Roles</strong></summary>

| Notebook               | Purpose                                                                                                       |
| ---------------------- | ------------------------------------------------------------------------------------------------------------- |
| `EDA.ipynb`            | Dataset exploration, channel analysis, mask analysis, RGB visualization, statistics, and correlation analysis |
| `prepro&model.ipynb`   | Baseline U-Net using all 12 raw channels                                                                      |
| `prepro&model_4.ipynb` | Feature engineering, ablation, and greedy channel search                                                      |
| `final_model.ipynb`    | Final Model Experiment 1 with an independent test set                                                         |
| `final_model_2.ipynb`  | Final Model Experiment 2 using an 80/20 train-validation split                                                |

</details>

---

## ⚠️ Limitations

* The dataset contains only 306 image-mask pairs.
* Performance can vary across scenes and split configurations.
* Some scenes contain no water.
* The models were trained from scratch without a pretrained encoder.
* Threshold selection was performed on validation data.
* The second final-model experiment does not contain an independent test set.
* Metrics from different evaluation protocols are not directly equivalent.

---

## 🚀 How to Run

### Install Dependencies

```bash
pip install numpy pandas rasterio scikit-learn matplotlib seaborn pillow torch
```

A CUDA-capable GPU is recommended for training.

### Workflow

Run the notebooks in the following order:

```text
EDA.ipynb
      ↓
prepro&model.ipynb
      ↓
prepro&model_4.ipynb
      ↓
final_model.ipynb
      ↓
final_model_2.ipynb
```

Update the dataset path used by the notebooks before training.

The project uses:

```text
SEED = 42
```

for reproducibility.

---

## 🛠️ Tech Stack

```text
Python
PyTorch
Rasterio
NumPy
Pandas
Scikit-learn
Matplotlib
Seaborn
Pillow
```

---

## 👤 Author

**Mahmoud Shoaib**

AI Engineer

[GitHub](https://github.com/mahmoudshoip94)
[LinkedIn](https://www.linkedin.com/in/mahmoud-shoaib-7040ba300/)

---

## 📌 Project Summary

This project explores water segmentation from multispectral satellite data through a complete experimental pipeline:

```text
Satellite Data
      ↓
EDA
      ↓
Preprocessing
      ↓
NDWI Baseline
      ↓
U-Net Baseline
      ↓
Feature Engineering
      ↓
Channel Selection
      ↓
Final Experiments
      ↓
Independent Evaluation
```

The repository documents the complete progression from a classical spectral baseline to deep-learning segmentation experiments, while keeping each evaluation protocol explicitly separated.
