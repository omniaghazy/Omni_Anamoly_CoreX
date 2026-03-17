# Walkthrough: OmniAnomaly CoreX V2 Completion

All objectives for the OmniAnomaly CoreX V2 project have been successfully met. The model is now robustly compatible with TensorFlow 1.15.5, the training and scoring issues have been resolved, and comprehensive documentation has been added.

## 1. Key Technical Fixes

### A. TensorFlow Double-Nesting & Import Fixes
The project was plagued by `AttributeError: module 'tensorflow._api.v1.compat.v1.compat' has no attribute 'v1'` due to an incorrect import pattern. I swapped all `tf.compat.v1.X` to `tf.X` as the global `tf` was already pointing to the compat layer. 

I also fixed the `ModuleNotFoundError` for `legacy_rnn` by using `tf.nn.rnn_cell` directly in `wrapper.py`.

### B. The Scoring Phase "ValueError"
We fixed a critical `ValueError` that occurred during the switch from training to scoring. The weights were being recreated instead of reused. By adding `reuse=tf.AUTO_REUSE` to the specific variable scopes wrapping the `mean_layer` and `std_layer` executions, the model now correctly shares weights across phases.

## 2. Documentation Added

- **RUNNING_GUIDE_V2.md**: Detailed environment and setup guide.
- **ARCHITECTURE_V2_OVERVIEW.md**: Explanation of the Hybrid GRU + Association Discrepancy architecture.

## 3. GitHub Deployment

The following components were pushed to the `main` branch of [omniaghazy/Omni_Anamoly_CoreX](https://github.com/omniaghazy/Omni_Anamoly_CoreX):
- All updated model source files.
- New V2 documentation files.
- Causal adjacency matrix (`causal_adj_matrix.npy`).
- Independent Kaggle standalone scripts and notebooks.

## 4. Final Validation Results

A full 1-epoch training run was executed. The results confirmed:
- **Phase 1 (Training)**: Successfully completed without crashes.
- **Phase 2 (Scoring)**: Correctly reuses weights and generates anomaly scores for both training and test sets.
- **Phase 3 (Evaluation)**: Successfully computed BF-Search thresholds and F1-metrics.
