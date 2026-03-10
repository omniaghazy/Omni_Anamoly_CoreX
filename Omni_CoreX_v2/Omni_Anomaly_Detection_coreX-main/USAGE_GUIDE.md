# OmniAnomaly: End-to-End Usage Guide (A to Z)

This guide walks you through the complete pipeline, from raw sensor data to final anomaly detection results.

---

## Phase 1: Data Preparation & Ingestion

### Step 1: Fix and Port Raw Data
Before preprocessing, we need to ensure the raw data is in the correct format and location.
- **Script**: `fix_data.py`
- **Action**: Opens your raw file (CSV or Excel) and converts it to a standardized `all_data.csv` inside `data/RobotArm/`.
- **Command**:
  ```powershell
  python fix_data.py
  ```
  > [!NOTE]
  > Edit the `RAW_DATA_SOURCE` variable in `fix_data.py` if your raw file name changes.

### Step 2: Clean and Scale Data
Once the raw CSV is ready, we need to handle missing values, select features, and apply normalization.
- **Script**: `data_preprocess.py`
- **Action**: Applied Z-score normalization (`StandardScaler`), handles time intervals, and generates the processed `.pkl` files needed for training.
- **Command**:
  ```powershell
  python data_preprocess.py
  ```

---

## Phase 2: Model Training

### Step 3: Run Training Loop
Now that the data is prepared in `data/processed/`, you can train the model.
- **Script**: `main.py`
- **Action**: Builds the SRNN + VAE architecture and minimizes the ELBO loss over multiple epochs.
- **Recommended Command (Optimized Settings)**:
  ```powershell
  python main.py --max_epoch=200 --z_dim=64 --window_length=120
  ```
  - `max_epoch`: How many times the model sees the full dataset.
  - `z_dim`: The complexity of the latent space (64 is recommended for 36+ sensors).
  - `window_length`: The temporal context (120 points).

---

## Phase 3: Evaluation & Results

### Step 4: Analyze Scores
After training, the model calculates anomaly scores (log-probabilities) for the test set.
- **Location**: `results/RobotArm_coreX_v2_optimized/`
- **Files**:
  - `config.json`: The hyperparameters used.
  - `test_score.pkl`: The raw anomaly scores.
  - `run_log.txt`: The final F1, Precision, and Recall metrics.

### Step 5: Visualize (Optional)
To see the anomalies visually vs. the reconstruction:
- **Script**: `plot_results.py`
- **Command**:
  ```powershell
  python plot_results.py
  ```

---

## Summary of the "A to Z" Pipeline
1.  **Ingest**: `python fix_data.py`
2.  **Clean**: `python data_preprocess.py`
3.  **Train**: `python main.py --max_epoch=200`
4.  **Review**: Check `results/` for the F1-Score report.

> [!TIP]
> Always ensure you are working within the `corex_env` conda environment to maintain dependency compatibility.
