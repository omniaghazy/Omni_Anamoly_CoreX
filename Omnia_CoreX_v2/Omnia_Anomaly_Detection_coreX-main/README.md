# OmniAnomaly for Anomaly Detection (Robot Arm)

This repository contains an optimized version of the **OmniAnomaly** model, a Stochastic Recurrent Neural Network (SRNN) for multivariate anomaly detection in time series data. Specifically, it has been tuned for robot arm sensor telemetry.

## 🚀 Key Features
- **Temporal Modeling**: GRU-based recurrent architecture.
- **$\beta$-VAE Loss**: Controlled reconstruction/regularization balance for improved F1 scores.
- **LGSSM Transition**: Linear Gaussian State Space Model for latent evolution.
- **Planar Normalizing Flows**: Enhanced posterior expressiveness.
- **Standardized Scaling**: Robust data preprocessing with `StandardScaler`.

## 📂 Project Structure
- `omni_anomaly/`: Core model components (VAE, Recurrent Distributions, etc.)
- `main.py`: Main entry point for training and evaluation.
- `data_preprocess.py`: Data cleaning and feature engineering.
- `requirements.txt`: Environment dependencies (TF 1.15.x compatible).

## 🛠️ Usage

### 1. Preprocess Data
```powershell
python data_preprocess.py
```

### 2. Train Model
```powershell
python main.py --max_epoch=200 --z_dim=64 --window_length=120
```

### 3. Evaluate results
Results are automatically saved in the `results/` directory, including F1, Precision, and Recall metrics.

## 📖 Methodology
For a detailed scientific breakdown of the model, architecture, and recent optimizations, please refer to the `omni_anomaly_scientific_report.md` artifact.
