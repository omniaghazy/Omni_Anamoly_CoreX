# Omnia CoreX: Complete File & Structure Guide

This guide explains the purpose of every file and folder in the Omnia CoreX project so anyone can understand the architecture at a glance.

## 📂 Root Directory / General Workflow
- **`zip_for_kaggle.py`**: A utility script used to package the V2 project securely into a `.zip` file for uploading to Kaggle as a dataset. It explicitly ignores large, unnecessary folders like `__pycache__` and existing zip files to save bandwidth.

---

## 📂 Project Directory Breakdown (Applies to `Omnia_Anomaly_Detection_coreX-main/` in V1 and V2)

### 1. Data Processing Scripts
- **`data_preprocess.py`**: The first script you should run. It cleans raw data (like `rtde_data.csv`), normalizes the values, and creates standard train/test subsets stored as `.pkl` files in the `data/` folder.
- **`feature_engineering.py` (V2 only)**: Runs after `data_preprocess.py`. It extracts deeper temporal insights (like rolling averages) to feed a "smarter" dataset into the model.
- **`fix_data.py`**: A utility script to repair corrupted or fundamentally misaligned datasets if the initial extraction fails.
- **`explore_data.py` & `inspect_pkl.py`**: Debugging tools to visually print and inspect the contents of the generated `.pkl` data files without running the entire model.

### 2. Model & Execution Scripts
- **`main.py`**: The core execution hub. It loads the preprocessed data, builds the OmniAnomaly neural network graph, and begins the training loop. Modify parameters like epochs, batch sizes, and learning rates here.
- **`anomaly_factory.py`**: Contains the factory classes and core algorithmic logic for how anomalies are formally scored, defined, and evaluated once the model executes.
- **`check_gpu.py`**: A diagnostic tool to verify if your TensorFlow 1.15 environment successfully detects your NVIDIA GPU.

### 3. Results & Visualization
- **`plot_results.py`**: Reads the output anomaly scores and draws comparative threshold graphs to visually represent where the robot arm failed.
- **`quick_results.py` (V2 only)**: A fast, lightweight script to print a text-based summary of F1-scores and peak anomalies without drawing full graphs.

### 4. Execution Shortcuts (Batch files)
- **`run_all.bat` (V2 only)**: A single-click Windows execution file that sequentially runs data processing, feature engineering, model training, and plotting in one go.
- **`run_preprocess.bat`**: Shortcut to execute just the data preprocessing phase.
- **`run_train.bat`**: Shortcut to execute just the model training phase.

### 5. Core Libraries & Folders
- **`omni_anomaly/`**: The local library directory containing the custom neural network architectures, GRU configurations, and loss functions. This relies on the open-source Zhusuan Bayesian network package. You rarely need to touch this folder.
- **`data/`**: The output folder where all generated `.pkl` (Pickle format) datasets and intermediate inputs are stored.
- **`results/`**: The output folder where trained models, evaluation logs, resulting CSV subsets, and PNG plots are saved post-training.
- **`model_coreX_v1/`**: Directory tracking the saved model checkpoints to pause and resume training.

### 6. Documentation & References
- **`requirements.txt`**: The exact pip dependencies required to establish the Python environment.
- **`Colab_VSCode_Guide.md` & `SSH_Success_Workflow.md`**: Guides detailing alternative execution strategies, like how to establish a tunnel into a Cloud instance via VSCode SSH.
- **`USAGE_GUIDE.md` (V2)**: Expanded instructions detailing execution arguments.
- **`README.md`**: A brief introductory file.
