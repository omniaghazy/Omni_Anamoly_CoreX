# Version 1: Baseline OmniAnomaly

This folder contains the **V1 Baseline** implementation of the OmniAnomaly model. It is completely isolated from V2 and contains its own processed data and trained checkpoints.

## Setup

Ensure you have installed the requirements:
```bash
pip install -r requirements.txt
```

*(Note: TensorFlow 1.x / 2.x compatibility is handled automatically by the codebase.)*

## Running the Code

1. **Preprocess Data (if needed)**
   The `data/processed/` folder already contains the required `.pkl` files.
   If you need to re-run preprocessing from the raw `.csv`/`.xlsx` files:
   ```bash
   python data_preprocess.py
   ```

2. **Train the Model & Evaluate**
   To run the complete pipeline (training + scoring):
   ```bash
   python main.py
   ```

## Key Files

- `main.py`: The entry point for training and evaluation.
- `model_coreX_v1/`: Contains the pre-trained checkpoints.
- `data/`: Contains the `RobotArm` raw data and `processed/` .pkl files.
- `omni_anomaly/`: The core model architecture.
- `results/`: Output folder for final evaluation metrics and plots.
