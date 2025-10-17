

# Copilot Instructions for Lesion_Mapping v1.2 (Stroke Segmentation)

## Scope
These instructions apply only to the contents of `stroke_segmentation_v1.2/`. Ignore older versions and other folders.

## Project Overview
This folder implements deep learning for 3D stroke lesion segmentation using full-resolution MRI data. The architecture and workflows are optimized for medical imaging, TensorFlow 2.15.1, and large volumetric datasets.

## Architecture & Data Flow
- **Training Notebooks**: Use `stroke_seg_v1.2_training.ipynb` and `stroke_seg_v1.2_training_with_denoise.ipynb` for model training. These contain all code for data loading, preprocessing, augmentation, model definition, and training loops.
- **Testing Notebook**: Use `stroke_seg_v1.2_testing.ipynb` for model evaluation and inference.
- **Callbacks**: Custom callbacks and metrics are in `callbacks/dynamic_production/`.
- **Model Storage**: Trained models are saved as `.keras` files in `models/dynamic_production/`.
- **Config Files**: Training and model configs are in `models/dynamic_production/config.json`.
- **Logs**: Training/debug logs are in `logs/`.
- **Scripts**: Any batch or SLURM scripts are in `scripts/` (if present).

## Developer Workflows
- **Training**: Run the training notebook(s) interactively. For reproducible runs, use the config file in `models/dynamic_production/config.json`.
- **Testing**: Use the testing notebook to evaluate models and generate metrics.
- **Model Loading**:
  ```python
  from tensorflow.keras.models import load_model
  model = load_model('models/dynamic_production/smart_sota_dynamic_20250918_152508.keras')
  ```
- **Data Preprocessing**: Use `nibabel` for NIfTI MRI files. Standard normalization and optional denoising are implemented in the training notebooks.

## Project-Specific Patterns
- **Full-Resolution Processing**: All MRI volumes are processed at native resolution; do not crop or resize unless explicitly specified.
- **Custom Augmentation**: Augmentation logic is embedded in the training notebooks and may differ from standard Keras/TensorFlow approaches.
- **Single-GPU Preference**: Training is optimized for single GPU; multi-GPU setups may cause instability.
- **Memory Optimization**: Batch size and layer design are tuned for large 3D volumes. Monitor GPU memory usage.
- **Metrics**: Validation metrics and thresholds are stored in `callbacks/dynamic_production/` as JSON files.

## External Dependencies
- TensorFlow 2.15.1
- Python 3.11
- nibabel, numpy, scipy

## Conventions & Tips
- **Configurable Parameters**: Use `models/dynamic_production/config.json` for experiment reproducibility.
- **Logs**: Check `logs/` for training/debug info.
- **MRI Shape**: Standard input shape is determined by the dataset; do not assume fixed dimensions.
- **Best Model**: The best model checkpoint is `callbacks/dynamic_production/best_model_dynamic.keras`.

## References
- See the training/testing notebooks for code patterns and workflow details.
- Metrics and calibration files are in `callbacks/dynamic_production/`.

---
If any section is unclear or missing, provide feedback or point to specific files for deeper analysis.