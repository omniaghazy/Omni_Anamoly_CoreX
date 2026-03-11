# Changelog: Version 1 vs Version 2

This document outlines the major differences, improvements, and fixes that occurred between **Version 1** and **Version 2** of the Omni CoreX anomaly detection project.

## Version 1 (V1)
V1 served as the baseline implementation, adapting the OmniAnomaly framework to the robot manipulation data (`rtde_data.csv`).
- **Data Preprocessing**: Basic ingestion of `rtde_data.csv`, filtering basic operational columns, normalizing data, and packing into `.pkl` format for the model (`data_preprocess.py`).
- **Core Model**: Integrated Zhusuan and TFSnippet within a TensorFlow 1.15 environment.
- **Training Pipeline**: Conducted via `main.py`, providing foundational anomaly scores reconstruction.
- *Limitations*: Preprocessing in V1 was minimal, potentially leaving noisy features, and the pipeline required manual successive script executions. Model training times were exceptionally long.

## Version 2 (V2)
V2 introduced significant architectural enhancements to both the data pipeline and execution workflow aimed at better accuracy and usability.

### 1. Advanced Feature Engineering (`feature_engineering.py`)
- Added a dedicated feature engineering script.
- Introduced sliding windows, moving averages, and advanced statistical features to better capture temporal anomalies in the robot arm's movement.

### 2. Upgraded Preprocessing (`data_preprocess.py`)
- Refined the feature selection process to heavily penalize missing values and filter out irrelevant columns early on.
- Optimized the way train/test splits are assembled.

### 3. Pipeline Automation (`run_all.bat`)
- Added a unified `run_all.bat` script that automatically chains preprocessing, feature engineering, model training, and plotting without manual intervention.

### 4. Code Structuring & Reporting Tools
- Added `plot_results.py` and `quick_results.py` in V2 to rapidly extract anomaly scores and plot comprehensive charts (e.g., `RobotArm_final_report.png`) without having to parse raw terminal text.
- Included `USAGE_GUIDE.md` for specific local execution instructions.

### 5. TF2 & Causal Graph Integration (Latest Update)
- Integrated the `CausalGraphModule` using spatial analysis.
- Introduced `standardize_tf_imports.py` to transparently map legacy TF 1.x `tf.contrib` and GRU module calls to modern TF 2.x Keras equivalencies without breaking the original architecture.
- Added a native TF 2.x / Keras 3 Kaggle notebook (`corex-v2-causal-graph-kaggle.ipynb`) to bypass Python 3.6 Conda environment requirements entirely.

## Kaggle Fixes Applied (During V2)
Because V2 demands substantial compute power, we formulated fixes specifically to deploy V2 via Kaggle Notebooks.
- **Environment Resolution**: Solved Kaggle compatibility issues with TF1.15 by constructing scripts to install a custom Miniconda Python 3.6 environment on top of Kaggle's native image.
- **Dependency Tracking**: Directly resolved `tfsnippet` and `zhusuan` installation issues by targeting specific repository branches instead of using broken pip distributions.
- **Zipping Utility**: Created `zip_for_kaggle.py` in the root folder to securely package V2 while ignoring massive cache files (`__pycache__`), keeping the upload lightweight and efficient for Kaggle dataset creation.
- **Epoch & Code patching**: Deployed patches to `main.py` directly on Kaggle to handle shorter/test epochs properly.
