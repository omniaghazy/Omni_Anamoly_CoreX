# Version 2: Hybrid Causal Graph OmniAnomaly

This folder contains the **V2 Optimized** implementation, which integrates a spatial **Causal Graph Module** directly into the OmniAnomaly architecture. It is completely isolated from V1 and contains its own processed data, causal adjacency matrix, and trained checkpoints.

## Setup

Ensure you have installed the requirements:
```bash
pip install -r requirements.txt
```

*(Note: TensorFlow 1.x / 2.x compatibility is handled automatically by the codebase.)*

## Architecture Upgrades over V1

- **Causal Graph Integration**: Uses `causal_adj_matrix.npy` to spatially weight variables during reconstruction.
- **Improved Data Windowing**: Different sliding window size and feature set compared to V1.
- **Handling of TF Scopes**: Fixes applied for variable reuse (`tf.AUTO_REUSE`) to allow seamless training and scoring in a single execution.

## Running the Code

1. **Preprocess Data (if needed)**
   The `data/processed/` folder already contains the required `.pkl` files (which are different from V1).
   If you need to re-run preprocessing:
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
- `causal_adj_matrix.npy`: The Causal Graph structural matrix.
- `model_coreX_v2_optimized/`: Contains the pre-trained checkpoints for V2.
- `data/`: Contains the raw data and `processed/` .pkl files specific to V2.
- `omni_anomaly/`: The customized model architecture with Graph operations.
- `results/`: Output folder for final evaluation metrics.
