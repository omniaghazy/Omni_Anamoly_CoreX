# OmniAnomaly CoreX V2: Running Guide

This guide explains how to set up the environment and run the "massive" OmniAnomaly CoreX V2 model.

## 1. Environment Compatibility (CRITICAL)

The model is built for **TensorFlow 1.15.5** and **Python 3.7**. Newer versions of TensorFlow (2.x) or Python (3.13+) will fail due to legacy dependency constraints and C++ DLL incompatibilities.

### Recommended Setup:
- **OS**: Windows (tested) / Linux
- **Conda Environment**: `corex_env`
- **Python**: 3.7.16
- **TensorFlow**: 1.15.5

## 2. Installation Steps

### Step A: Create the Conda Environment
```bash
conda create -n corex_env python=3.7.16 -y
conda activate corex_env
```

### Step B: Install Core Dependencies
The version of `PyYAML` must be older than 6.0 to avoid breaking the parameter parser in `tfsnippet`.
```bash
pip install tensorflow==1.15.5
pip install "pyyaml<6.0"
pip install numpy==1.19.5 scipy scikit-learn pandas matplotlib
```

### Step C: Install Local Patched Libraries
This project relies on modified versions of `tfsnippet` and `zhusuan`. Do **NOT** install them from pip.
```bash
# Example paths - replace with your local paths
pip install -e c:\users\memom\omnianomaly\tfsnippet
# If zhusuan is also local:
# pip install -e c:\users\memom\omnianomaly\zhusuan
```

## 3. Data Preparation

Place your `.pkl` files (e.g., `RobotArm_train.pkl`) in the following directory structure:
```text
data/
└── processed/
    ├── RobotArm_train.pkl
    └── RobotArm_test.pkl
```

## 4. Running the Model

### Training (1 Epoch Test)
Use the `--max_epoch 1` flag to verify that everything is working.
```bash
python main.py --max_epoch 1
```

### Full Training
```bash
python main.py --max_epoch 100
```

## 5. Troubleshooting Common Errors

### `AttributeError: ... has no attribute 'v1'`
This usually means you are attempting to run in a TF 2.x environment. Ensure `corex_env` is active.

### `ValueError: Variable ... already exists`
This was fixed in V2 by adding `tf.AUTO_REUSE` to the `wrapper.py` logic. If encountered, ensure your `omni_anomaly/wrapper.py` is up to date.

### `TypeError: load() missing 1 required positional argument: 'Loader'`
Root Cause: PyYAML 6.0+. 
Fix: `pip install "pyyaml<6.0"`.
