# Master Project Guide: OmniAnomaly for CoreX

Welcome to the **OmniAnomaly CoreX** repository! This project implements an advanced deep learning framework for multivariate time-series anomaly detection, specifically tailored for the **CoreX** task (such as monitoring complex systems like Robot Arms).

Due to the evolution of the project architecture, the codebase has been massively restructured into two completely standalone, isolated versions to ensure readability, reproducibility, and prevent cross-contamination of models or data.

---

## 🏗️ The Two Versions: Complete Isolation

### 1. `Version_1_Baseline/`
This is the **original, unmodified baseline** OmniAnomaly model.
- **Purpose:** Serve as the foundation and benchmark for evaluating future architectural improvements.
- **Architecture:** Standard OmniAnomaly (Stochastic Recurrent Neural Network with Normalizing Flows).
- **Data:** Uses its own specifically windowed `data/processed/` dataset. 

### 2. `Version_2_Hybrid/`
This is the **final, optimized hybrid architecture**.
- **Purpose:** Provide a significantly upgraded anomaly detection capability.
- **Key Upgrades:**
  - **Causal Graph Module:** Integrates a spatial adjacency matrix (`causal_adj_matrix.npy`) to explicitly model the relational dependencies between different sensors/variables.
  - **Y-Split & Association Discrepancy:** Advanced techniques to better reconstruct expected vs. observed behaviors.
  - **GRU (Gated Recurrent Unit):** Optimized temporal processing.
- **Data:** Uses a different sliding window size and feature set compared to V1.

---

## 📂 Repository Structure

To keep the workspace pristine, everything has a place:

```text
/
├── Version_1_Baseline/     # The V1 codebase, data, and pre-trained checkpoints
├── Version_2_Hybrid/       # The V2 codebase, data, and pre-trained checkpoints
├── notebooks/              # Google Colab and Kaggle notebooks for cloud execution
├── archives/               # Zipped snapshots ready for Kaggle/Colab uploads
├── debug_logs/             # 🗑️ Legacy documentation, test scripts, and debug crash logs
└── README_GLOBAL.md        # You are reading this!
```

---

## 🚀 Quick-Start Instructions

Because the two versions are completely separated, running the code is incredibly simple. You must navigate into the folder of the version you want to use.

### Running Version 1 (Baseline)
```bash
cd Version_1_Baseline
python main.py
```

### Running Version 2 (Hybrid Causal)
```bash
cd Version_2_Hybrid
python main.py
```

*Note: The datasets are already preprocessed and included in each version's respective `data/processed/` folder. The checkpoints are also pre-trained and saved, so `main.py` will execute the full pipeline (training phase + scoring phase) instantly.*
