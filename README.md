# Chest X-Ray Augmentation & Model Selection Project

This repository contains scripts used to filter and organize a Chest X-Ray dataset, apply and test image augmentation strategies, and run model experiments to find the best performing architecture for a small set of diagnostic classes (e.g., `Mass`, `Nodule`, `No Finding`). The README summarizes dataset layout, augmentation techniques, experiment reproduction steps, dependencies, and where outputs are stored.

**Quick overview**
- **Source scripts**: `lung cancer image organize.py`, `data_preprocessor.py` (dataset creation)
- **Augmentation code**: `dataOgmentation.py`, `augmentation_switch.py`, `test_augmentations.py`
- **Model utils**: `model_utils.py`, model definitions in `densenet121.py`, `densenet169.py`, `densenet201.py`, `resnet50.py`, `mobilenet_v2.py`, `VGG16.py`, `efficientnet_b3.py`
- **Outputs**: `Trained_Models/`, `Model_Predictions/`, `runs/` (TensorBoard logs), `Augmentation_Samples/`

**Repository layout (relevant files & folders)**
- `Filtered_CXR_Dataset/` (expected input to `data_preprocessor.py`): per-class image folders (e.g., `Mass/`, `Nodule/`, `No Finding/`) and `filtered_metadata.csv`.
- `data_preprocessor.py`: samples and splits the `Filtered_CXR_Dataset` into a model-ready dataset (`CRX_Model_Dataset/`) and creates `training_metadata.csv` and `testing_metadata.csv`.
- `dataOgmentation.py`: implements augmentations, utilities to save examples, TensorBoard logging, and interactive visualization helpers.
- `augmentation_switch.py`: global switches for enabling/disabling augmentation types.
- `test_augmentations.py`: test harness that runs the augmentation workflows and logs to TensorBoard.
- `model_utils.py`: helper utilities for checking paths, creating output folders and TensorBoard writers.

**Augmentation techniques implemented**
- Histogram equalization (applied to grayscale images before further transforms).
- Grayscale -> 3-channel conversion for downstream models expecting RGB input.
- Horizontal flip: controlled by `AUGMENTATION_SWITCHES['use_horizontal_flip']`.
- Rotation: controlled by `AUGMENTATION_SWITCHES['use_rotation']`; class-specific default rotation degrees are set inside `dataOgmentation.py`.
- Zoom / RandomResizedCrop: controlled by `AUGMENTATION_SWITCHES['use_zoom_and_crop']` (scale ranges configurable per-class).
- Perspective distortion: controlled by `AUGMENTATION_SWITCHES['use_perspective']`.
- Gaussian noise: implemented as a custom `AddGaussianNoise` transform, controlled by `AUGMENTATION_SWITCHES['use_gaussian_noise']`.

Class-aware intensity defaults are set in `dataOgmentation.py` and can be overridden via the `intensity_controls` dict (see examples in the `__main__` blocks of `dataOgmentation.py` and `test_augmentations.py`).

**Where results and artifacts go**
- TensorBoard logs: `runs/` (e.g., `runs/augmentation_demo`, `runs/augmentation_test`, `runs/<model_name>`).
- Saved augmentation visuals: `augmentation_tests/`, `augmented_samples/`, and `Augmentation_Samples/` (script-created examples).
- Trained model artifacts: `Trained_Models/` (e.g., `densenet121_best.pth` in the repo).
- Model predictions and reports: `Model_Predictions/` and `Trained_Models/*_classification_report.json`.

Usage and reproduction
----------------------
1) Prepare the filtered dataset
- Place your pre-filtered Chest X-Ray dataset under the `Filtered_CXR_Dataset/` folder with subfolders named per-class (e.g., `Mass`, `Nodule`, `No Finding`) and ensure a `filtered_metadata.csv` file exists alongside the folders. `data_preprocessor.py` expects these locations by default (variables are at top of the file).

2) Create a model-ready dataset (sample & split)
- Edit the constants at the top of `data_preprocessor.py` (paths, `TARGET_CATEGORY_COUNTS`, `TRAIN_SPLIT_RATIO`, `RANDOM_SEED`) if needed, then run:

```bash
python data_preprocessor.py
```

This will create `CRX_Model_Dataset/Training_set/` and `CRX_Model_Dataset/Testing_set/` with per-class folders and `training_metadata.csv` / `testing_metadata.csv`.

3) Inspect and tune augmentations
- Toggle augmentation switches in `augmentation_switch.py`.
- Use `test_augmentations.py` to generate images and TensorBoard logs for quick inspection:

```bash
python test_augmentations.py
# then view with
tensorboard --logdir=runs
```

You can also call `dataOgmentation.visualize_and_adjust` in a notebook to iteratively tune `intensity_controls`.

4) Train models
- Model definitions exist for common architectures (`densenet*`, `resnet50`, `VGG16`, `mobilenet_v2`, `efficientnet_b3`) but there is no single centralized training CLI in the repo yet. Use these guidelines:
  - Ensure `CRX_Model_Dataset` exists and `model_utils.check_paths()` passes.
  - Adapt one of the model files (e.g., `densenet121.py`) to implement the training loop and use `torch.utils.data.ImageFolder` or a custom `Dataset` that uses `training_metadata.csv`.
  - Log training metrics to TensorBoard using `SummaryWriter(f'runs/{model_name}')`.
  - Save checkpoints to `Trained_Models/` and predictions to `Model_Predictions/`.

5) Evaluate and compare
- After training, export classification reports into `Trained_Models/` and prediction CSVs into `Model_Predictions/`. Use those outputs to compare model performance across architectures and augmentation strategies.

Dependencies
------------
- Python 3.8+ (tested with 3.8–3.10)
- PyTorch and torchvision
- PIL/Pillow
- pandas
- tqdm
- tensorboard
- numpy

Suggested install (adjust versions to your environment):

```bash
pip install torch torchvision pillow pandas tqdm tensorboard numpy
```

Notes & suggestions
-------------------
- The augmentation pipeline uses PIL-based transforms; models typically expect 3-channel inputs (the repo converts grayscale to 3-channel images).
- `augmentation_switch.py` makes it easy to run ablation studies by toggling augmentation components on/off. For reproducible experiments, track the switch values and `intensity_controls` used for each run.
- If you have large binary assets in Git history, consider `git-lfs` for storing models or large images.
- If you'd like, I can:
  - Add a `requirements.txt` or `environment.yml`.
  - Create a simple training script that uses `ImageFolder` and the augmentation pipeline.
  - Generate a short script to run augmentation ablation experiments and save results CSVs.
