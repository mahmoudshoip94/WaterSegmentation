# Water Segmentation Using Multispectral and Optical Data

A deep-learning semantic segmentation project for detecting water pixels from 12-channel multispectral, optical, terrain, land-cover, and water-occurrence data. The model is a U-Net implemented from scratch in PyTorch, and the project also includes a classical NDWI threshold baseline for comparison.

---

## Project Summary

* **Task:** Binary water segmentation.
* **Input:** 12-channel raster data, with optional spectral indices such as NDWI.
* **Masks:** `0 = Background`, `1 = Water`.
* **Main model:** U-Net implemented from scratch in PyTorch.
* **Baseline:** Classical NDWI thresholding.
* **Development:** Baseline U-Net → Feature Engineering + Channel Selection → Final Model Experiment 1 → Final Model Experiment 2.

The project covers dataset exploration, preprocessing, feature engineering, classical baseline evaluation, U-Net training, threshold tuning, experiment comparison, and final model evaluation.

---

## Dataset

The dataset consists of multispectral raster images and binary masks.

| Item                        |                         Value |
| --------------------------- | ----------------------------: |
| Multispectral `.tif` images |                           306 |
| Binary `.png` masks         |                           306 |
| Image size                  |                     128 × 128 |
| Input channels              |                            12 |
| Mask values                 | `0 = Background`, `1 = Water` |
| Images containing water     |                           261 |
| Images with no water        |                            45 |
| Total water pixels          |                     1,302,272 |
| Total background pixels     |                     3,711,232 |
| Water percentage            |                        25.98% |
| Background percentage       |                        74.02% |

### Channel Order

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

For visualization:

* Red = channel 3
* Green = channel 2
* Blue = channel 1

### Categorical Channels

The following channels are treated as categorical and are **not normalized**:

* QA Band = channel 7
* ESA WorldCover = channel 10

The remaining 10 channels are continuous and are normalized.

### Invalid Value Handling

* MERIT DEM contains `-9999` values. These are treated as invalid and replaced using that channel’s mean.
* Normal negative spectral values were retained.
* Negative readings in spectral bands are not automatically treated as errors.
* The data are raster arrays used for image segmentation, not ordinary RGB images.

---

## Project Structure

The repository contains the following notebooks:

```text
WaterSegmentation/
├── data/
│   ├── images/
│   └── labels/
├── notebooks/
│   ├── EDA.ipynb
│   ├── prepro&model.ipynb
│   ├── prepro&model_4.ipynb
│   ├── final_model.ipynb
│   └── final_model_2.ipynb
├── figures/
└── README.md
```

### Notebook Roles

| Notebook               | Purpose                                                                                                       |
| ---------------------- | ------------------------------------------------------------------------------------------------------------- |
| `EDA.ipynb`            | Dataset exploration, channel analysis, mask analysis, RGB visualization, statistics, and correlation analysis |
| `prepro&model.ipynb`   | Baseline U-Net using all 12 raw channels                                                                      |
| `prepro&model_4.ipynb` | Feature Engineering, Ablation, and Greedy Channel Search                                                      |
| `final_model.ipynb`    | Final Model Experiment 1 with an independent test set                                                         |
| `final_model_2.ipynb`  | Final Model Experiment 2 using an 80/20 train-validation split with no independent test set                   |

---

## Exploratory Data Analysis

The EDA notebook examined image channels, masks, per-image statistics, and channel relationships.

Key observations:

* All masks contain only `0` and `1`.
* Water is present in 261 images and absent in 45 images.
* Water covers approximately 25.98% of all pixels.
* QA Band and WorldCover behave as categorical features.
* Several continuous channels have different distributions and ranges.
* DEM channels contain invalid `-9999` values that require handling.

### EDA Figures

![Figure — RGB composite](figures/eda_rgb_composite.png)

*RGB composite generated from Red, Green, and Blue channels.*

![Figure — Ground-truth water mask](figures/eda_label_mask.png)

*Example binary water mask. White = water, black = background.*

![Figure — 12 spectral bands](figures/eda_12_bands.png)

*Representative visualization of the 12 input channels for one sample.*

![Figure — Per-image channel min/max distributions](figures/eda_channel_minmax.png)

*Per-image minimum and maximum distributions across channels.*

![Figure — Correlation matrix between channels](figures/eda_correlation_matrix.png)

*Correlation matrix between all 12 channels.*

---

## Preprocessing

Preprocessing was implemented with `rasterio`, NumPy, and PyTorch.

### Main Steps

1. Read `.tif` images with `rasterio`.
2. Read binary `.png` masks.
3. Replace invalid MERIT DEM `-9999` values with the channel mean.
4. Preserve valid negative spectral values.
5. Compute dataset-wise, per-channel normalization statistics using only the training set.
6. Exclude QA Band and WorldCover from normalization.
7. Add spectral indices when enabled.
8. Apply optional augmentation only to the training set.
9. Use `SEED = 42` for reproducibility.

### Baseline Split

The Baseline U-Net in `prepro&model.ipynb` uses a stratified 70/15/15 split:

| Split      | Images |
| ---------- | -----: |
| Train      |    214 |
| Validation |     46 |
| Test       |     46 |
| Total      |    306 |

The split preserves the water/no-water distribution across the three subsets.

### Final Model Split

`final_model.ipynb` uses a stratified 80/10/10 split:

| Split      | Images |
| ---------- | -----: |
| Train      |    244 |
| Validation |     31 |
| Test       |     31 |
| Total      |    306 |

`final_model_2.ipynb` uses a stratified 80/20 split:

| Split      | Images |
| ---------- | -----: |
| Train      |    244 |
| Validation |     62 |
| Test       |      — |
| Total      |    306 |

For `final_model_2.ipynb`, the training set contains 36 no-water images and the validation set contains 9 no-water images.

The metrics from these different splits and evaluation protocols should not be treated as directly equivalent.

---

## Feature Engineering

Spectral indices were explored during development.

### NDWI

```python
NDWI = (Green - NIR) / (Green + NIR + 1e-6)
```

Channel indices:

* Green = 2
* NIR = 4
* SWIR1 = 5

Other indices explored during development included:

* MNDWI
* NDMI
* Combinations of NDWI, MNDWI, and NDMI

### Feature Engineering and Channel Selection

The experiments in `prepro&model_4.ipynb` evaluated raw channels, added spectral indices, channel ablation, and greedy channel removal.

Stage 1 validation results:

| Configuration  | Validation IoU |
| -------------- | -------------: |
| Raw 12         |         0.6035 |
| + NDWI         |         0.6043 |
| + MNDWI        |         0.6019 |
| + NDWI + MNDWI |         0.6010 |

Ablation on 14 channels:

| Configuration                     | Validation IoU |
| --------------------------------- | -------------: |
| Remove ESA WorldCover             |         0.6238 |
| Remove Water Occurrence           |         0.6132 |
| Remove QA Band                    |         0.6081 |
| Remove MERIT DEM                  |         0.6060 |
| Remove Copernicus DEM             |         0.6050 |
| Baseline 14-channel configuration |         0.6010 |

Greedy Channel Search started from the 13-channel configuration consisting of the 12 raw channels plus NDWI and MNDWI, after removing ESA WorldCover.

Removing Coastal Aerosol improved validation IoU to `0.6305`. No further channel removal improved the result.

The resulting feature configuration was:

```text
Raw channels:
[1,2,3,4,5,6,7,8,9,11]

Added:
NDWI
MNDWI

Removed:
Coastal Aerosol
ESA WorldCover
```

Final number of input channels: **12**.

### Feature Engineering Final Retraining

The final retraining in `prepro&model_4.ipynb` reached its best validation checkpoint at epoch `40`.

Validation:

| Metric    |  Value |
| --------- | -----: |
| IoU       | 0.6453 |
| F1        | 0.7844 |
| Precision | 0.8075 |
| Recall    | 0.7626 |

Independent test evaluation:

| Metric    |  Value |
| --------- | -----: |
| Loss      | 0.8851 |
| IoU       | 0.6505 |
| F1        | 0.7882 |
| Precision | 0.8498 |
| Recall    | 0.7350 |
| Accuracy  | 0.9130 |

This notebook is the **Feature Engineering and Channel Selection experiment** and is separate from both final-model experiments.

---

## Classical NDWI Baseline

A simple rule-based baseline was implemented using raw Green and NIR values.

Threshold search:

```python
thresholds = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
```

The threshold was selected using the validation set only.

**Best validation threshold:** `-0.3`

### Classical NDWI Results

| Metric         |  Value |
| -------------- | -----: |
| Validation IoU | 0.7067 |
| Validation F1  | 0.8282 |
| Test IoU       | 0.5052 |
| Test F1        | 0.6712 |

The best NDWI threshold was selected on validation and then evaluated on the test set without re-tuning.

---

## Deep Learning Model

The main model is a U-Net implemented from scratch in PyTorch.

### Baseline U-Net

The Baseline U-Net is implemented in `prepro&model.ipynb` before feature engineering or channel selection.

| Setting                 |             Value |
| ----------------------- | ----------------: |
| Input channels          |   12 raw channels |
| Output channels         |                 1 |
| Base filters            |                 8 |
| Depth                   |                 6 |
| Parameters              |         7,783,073 |
| Loss                    | BCEWithLogitsLoss |
| Optimizer               |              Adam |
| Learning rate           |            `1e-4` |
| Batch size              |                 8 |
| Maximum epochs          |                60 |
| Early stopping patience |                10 |

The baseline uses:

* No NDWI
* No MNDWI
* No channel selection
* All 12 raw channels
* Horizontal flip
* Vertical flip
* Rot90
* Light numerical-channel noise of `0.05`

Baseline results:

* Training stopped at epoch `52`.
* Best Validation IoU = `0.6176`.

Final test evaluation:

| Metric    |  Value |
| --------- | -----: |
| Loss      | 0.3642 |
| Accuracy  | 0.9117 |
| Precision | 0.8428 |
| Recall    | 0.7365 |
| F1        | 0.7861 |
| IoU       | 0.6475 |

The Baseline U-Net is a reference experiment and is not one of the final-model experiments.

### Final Model Experiments

Both final-model notebooks use a U-Net from scratch with:

* 12 raw channels + NDWI = 13 input channels
* MNDWI disabled
* NDMI disabled
* Base filters = 32
* Depth = 5
* Dropout = 0.1
* Output = 1 channel
* BCE + Dice loss

The two final experiments differ in their split and evaluation protocol.

---

## Training

Training was performed in PyTorch.

### Baseline Training Settings

The Baseline U-Net uses BCEWithLogitsLoss only, Adam with learning rate `1e-4`, batch size `8`, maximum `60` epochs, and early stopping patience `10`.

No `pos_weight` or DiceLoss is used in the baseline.

### Final Model Experiment 1 — `final_model.ipynb`

| Setting                   |               Value |
| ------------------------- | ------------------: |
| Split                     |            80/10/10 |
| Train / Validation / Test |       244 / 31 / 31 |
| Input channels            |                  13 |
| Optimizer                 |                Adam |
| Learning rate             |              `1e-4` |
| Weight decay              |              `1e-4` |
| Batch size                |                   8 |
| Loss                      |          BCE + Dice |
| `pos_weight`              |          ≈ `2.8228` |
| Scheduler                 |   CosineAnnealingLR |
| Gradient clipping         |                 1.0 |
| Early stopping patience   |                  20 |
| Maximum epochs            |                 100 |
| Model selection           | Best validation IoU |

Training:

* Best Epoch = `14`
* Early stopping at Epoch `34`

### Final Model Experiment 2 — `final_model_2.ipynb`

| Setting                 |               Value |
| ----------------------- | ------------------: |
| Split                   |               80/20 |
| Train / Validation      |            244 / 62 |
| Independent Test        |                  No |
| Input channels          |                  13 |
| Optimizer               |                Adam |
| Learning rate           |              `1e-4` |
| Weight decay            |              `1e-4` |
| Batch size              |                   8 |
| Loss                    |          BCE + Dice |
| `pos_weight`            |          ≈ `2.8200` |
| Scheduler               |   ReduceLROnPlateau |
| Scheduler factor        |               `0.5` |
| Scheduler patience      |                 `5` |
| Minimum LR              |              `1e-7` |
| Gradient clipping       |                 1.0 |
| Early stopping patience |                  20 |
| Maximum epochs          |                 100 |
| Model selection         | Best validation IoU |

Training:

* Best Epoch = `47`
* Early stopping at Epoch `67`

Because `final_model_2.ipynb` has no independent test set, its reported metrics are validation metrics only.

---

## Threshold Tuning

Threshold tuning was performed on the validation set for the final-model experiments.

### Final Model Experiment 1 — `final_model.ipynb`

The best validation threshold was:

**`0.6`**

Validation results after threshold tuning:

| Metric    |  Value |
| --------- | -----: |
| IoU       | 0.7479 |
| F1        | 0.8558 |
| Precision | 0.9423 |
| Recall    | 0.7838 |
| Accuracy  | 0.9387 |

The independent test set was then evaluated separately.

| Metric   |  Value |
| -------- | -----: |
| Test IoU | 0.5206 |
| Test F1  | 0.6848 |

### Final Model Experiment 2 — `final_model_2.ipynb`

The best validation threshold was:

**`0.6`**

Validation results after threshold tuning:

| Metric    |  Value |
| --------- | -----: |
| Loss      | 0.7192 |
| IoU       | 0.6991 |
| F1        | 0.8229 |
| Precision | 0.8840 |
| Recall    | 0.7697 |
| Accuracy  | 0.9166 |

These are **validation results after threshold tuning**. Since the same validation set was used for threshold tuning and evaluation, and there is no independent test set, they are not test results.

---

## Experiment History

The development progression was:

```text
EDA
↓
Baseline U-Net
↓
Feature Engineering + Ablation + Greedy Channel Search
↓
Final Model Experiment 1 (80/10/10 + Independent Test)
↓
Final Model Experiment 2 (80/20 + No Independent Test)
```

### Main Experiments

| Experiment                              | Split    | Input                          | Validation IoU | Validation F1 | Test IoU | Test F1 | Notes                                      |
| --------------------------------------- | -------- | ------------------------------ | -------------: | ------------: | -------: | ------: | ------------------------------------------ |
| Baseline U-Net                          | 70/15/15 | 12 raw                         |         0.6176 |             — |   0.6475 |  0.7861 | `prepro&model.ipynb`                       |
| Feature Engineering + Channel Selection | 70/15/15 | 10 selected raw + NDWI + MNDWI |         0.6453 |        0.7844 |   0.6505 |  0.7882 | `prepro&model_4.ipynb`                     |
| Final Model Experiment 1                | 80/10/10 | 12 raw + NDWI                  |         0.7479 |        0.8558 |   0.5206 |  0.6848 | `final_model.ipynb`; threshold `0.6`       |
| Final Model Experiment 2                | 80/20    | 12 raw + NDWI                  |         0.6991 |        0.8229 |        — |       — | `final_model_2.ipynb`; no independent test |

The metrics above come from different experiments and, in some cases, different train/validation/test splits. Validation and test metrics from different protocols should not be treated as directly equivalent.

---

## Selected Final Model

The repository contains two final-model experiments rather than a single model reported under one evaluation protocol.

### Final Model Experiment 1 — `final_model.ipynb`

| Item                       |                  Value |
| -------------------------- | ---------------------: |
| Split                      |               80/10/10 |
| Input                      | 12 raw channels + NDWI |
| Best validation checkpoint |               Epoch 14 |
| Best validation threshold  |                    0.6 |
| Validation IoU             |                 0.7479 |
| Validation F1              |                 0.8558 |
| Test IoU                   |                 0.5206 |
| Test F1                    |                 0.6848 |

This experiment includes an independent test set. The test metrics were reported from the single final test evaluation.

### Final Model Experiment 2 — `final_model_2.ipynb`

| Item                       |                  Value |
| -------------------------- | ---------------------: |
| Split                      |                  80/20 |
| Input                      | 12 raw channels + NDWI |
| Best validation checkpoint |               Epoch 47 |
| Best validation threshold  |                    0.6 |
| Validation IoU             |                 0.6991 |
| Validation F1              |                 0.8229 |
| Independent Test Set       |                   None |

The results from `final_model_2.ipynb` are validation-only results after threshold tuning and should not be presented as test results.

---

## Results

The main documented results are separated by experiment.

### Classical NDWI Baseline

| Metric | Validation |   Test |
| ------ | ---------: | -----: |
| IoU    |     0.7067 | 0.5052 |
| F1     |     0.8282 | 0.6712 |

### Baseline U-Net — `prepro&model.ipynb`

| Metric | Validation |   Test |
| ------ | ---------: | -----: |
| IoU    |     0.6176 | 0.6475 |
| F1     |          — | 0.7861 |

### Feature Engineering + Channel Selection — `prepro&model_4.ipynb`

| Metric | Validation |   Test |
| ------ | ---------: | -----: |
| IoU    |     0.6453 | 0.6505 |
| F1     |     0.7844 | 0.7882 |

### Final Model Experiment 1 — `final_model.ipynb`

| Metric | Validation | Independent Test |
| ------ | ---------: | ---------------: |
| IoU    |     0.7479 |           0.5206 |
| F1     |     0.8558 |           0.6848 |

### Final Model Experiment 2 — `final_model_2.ipynb`

| Metric | Validation |
| ------ | ---------: |
| IoU    |     0.6991 |
| F1     |     0.8229 |

`final_model_2.ipynb` has no independent test set.

The reported results should be interpreted within their respective split and evaluation protocols rather than as one directly comparable leaderboard.

---

## Visualizations for GitHub

The following figures are recommended for the repository. Save them under `figures/` using the filenames below.

### Figure 1 — RGB + Ground Truth

![Figure 1 — RGB and Ground Truth](figures/fig1_rgb_gt.png)
*RGB composite and corresponding binary water mask.*

### Figure 2 — Multispectral Bands

![Figure 2 — Multispectral Bands](figures/fig2_multispectral_bands.png)
*Representative Green, Red, NIR, SWIR1, DEM, and Water Occurrence channels.*

### Figure 3 — NDWI Baseline

![Figure 3 — NDWI Baseline](figures/fig3_ndwi_baseline.png)
*NDWI map, ground truth, and thresholded prediction using threshold `-0.3`.*

### Figure 4 — U-Net Prediction

![Figure 4 — U-Net Prediction](figures/fig4_unet_prediction.png)
*RGB, ground truth, U-Net prediction, and overlay.*

### Figure 5 — Training Curves

![Figure 5 — Training Curves](figures/fig5_training_curves.png)
*Training loss, validation loss, validation IoU, and validation F1.*

### Figure 6 — NDWI vs U-Net

![Figure 6 — NDWI vs U-Net](figures/fig6_ndwi_vs_unet.png)
*Validation IoU and Test IoU comparison between classical NDWI and U-Net within the relevant experiment.*

### Figure 7 — Error Visualization

![Figure 7 — Error Map](figures/fig7_error_map.png)
*RGB, ground truth, prediction, and error map. Error map convention: Green = True Positive, Red = False Positive, Blue = False Negative, Black = True Negative.*

---

## Limitations

* Dataset size is small: 306 image-mask pairs.
* Validation and test performance can vary between runs and split configurations.
* Some images contain no water, which can affect per-scene metrics.
* Performance can vary substantially across scenes.
* The model is trained from scratch without a pretrained encoder.
* Threshold selection is based on the validation set and may not generalize to all scenes.
* `final_model_2.ipynb` does not contain an independent test set.
* Metrics from different split and evaluation protocols should not be treated as directly equivalent.

---

## How to Run

### Dependencies

Install the required Python packages:

```bash
pip install numpy pandas rasterio scikit-learn matplotlib seaborn pillow torch
```

A CUDA-capable GPU is recommended for training.

### Running the Project

1. Clone the repository.
2. Mount Google Drive or update the dataset path used by the notebooks.
3. Run the notebooks according to the development workflow:

   * `EDA.ipynb`
   * `prepro&model.ipynb`
   * `prepro&model_4.ipynb`
   * `final_model.ipynb`
   * `final_model_2.ipynb`
4. Checkpoints and normalization statistics are saved according to the configuration used by each notebook.

The project uses `SEED = 42` for reproducibility.

---

## Conclusion

This project implemented and evaluated a multispectral water segmentation pipeline using 12-channel raster data. A classical NDWI threshold baseline was evaluated alongside U-Net models trained from scratch in PyTorch.

The development process progressed from a Baseline U-Net using all 12 raw channels to feature engineering, channel ablation, and greedy channel search. The feature-engineering experiment produced a 12-channel configuration consisting of selected raw channels plus NDWI and MNDWI.

Two separate final-model experiments were then evaluated using 12 raw channels plus NDWI. `final_model.ipynb` uses an 80/10/10 split with an independent test set and reports Test IoU `0.5206` and Test F1 `0.6848`. `final_model_2.ipynb` uses an 80/20 split with no independent test set and reports Validation IoU `0.6991` and Validation F1 `0.8229` after threshold tuning.

All results are documented separately according to their respective notebook, split, and evaluation protocol.
