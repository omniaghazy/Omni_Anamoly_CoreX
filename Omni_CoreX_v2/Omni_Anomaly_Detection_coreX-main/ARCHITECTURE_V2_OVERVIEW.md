# OmniAnomaly CoreX V2: Architecture Overview

The CoreX V2 model is a **Hybrid VAE (Variational Auto-Encoder)** designed for multivariate time-series anomaly detection. It combines the sequential modeling power of Gated Recurrent Units (GRU) with a spatial-attentional mechanism called Association Discrepancy (AD).

## 1. High-Level Architecture

The model follows the standard VAE "Sandwich" structure but with enhanced branches for feature extraction.

### A. The Hybrid Encoder (q_z | x)
When raw time-series data $X$ enters the model, it splits into two parallel branches:

1.  **Branch A (Sequential - GRU)**:
    *   A bidirectional GRU processes the time window (length 120).
    *   Captures temporal dependencies: "How did sensor A behave 5 seconds ago?"
2.  **Branch B (Spatial/Relational - Association Discrepancy)**:
    *   Uses a multi-layer AD mechanism (3 layers).
    *   Processes the causal relationships between sensors (75 sensors total).
    *   Captures spatial dependencies: "If sensor A spikes, should sensor B also spike?"

**Fusion Strategy**: The outputs of both branches are concatenated to form a rich feature vector that represents the total state of the system at that timestamp.

### B. The Latent Space (z)
*   The fused features are projected into a Stochastic Latent Space ($z$).
*   **Planar Flows**: 20 layers of Normalizing Flows (NF) are applied to the posterior distribution $q(z|x)$ to transform it into a complex, multi-modal shape, allowing it to model very intricate sensor behaviors.

### C. The Decoder (p_x | z)
*   The decoder reconstructs the original sensor values from the latent sample $z$.
*   It uses a mirror-image GRU structure to ensure the reconstruction respects temporal continuity.

## 2. Mathematical Objective (Loss Function)

The model is optimized using the **ELBO (Evidence Lower Bound)** with spatial regularization:

$$Loss = Reconstruction\_Error + KL\_Divergence + AD\_Loss$$

- **AD Loss (Association Discrepancy)**: Minimizes the distance between "Prior Association" (learned base correlations) and "Series Association" (observed correlations in the current window). Large discrepancies here are strong indicators of structural anomalies.

## 3. Anomaly Scoring

Anomaly scores are computed using the **Reconstruction Probability**.

1.  During test/scoring, we sample $z$ multiple times (default `test_n_z=10`).
2.  We calculate the log-likelihood of observing the actual data given these samples.
3.  **BF-Search**: The system automatically searches for the optimal threshold that maximizes the F1-Score on the provided test labels.

## 4. Why this is "CoreX"?

The "CoreX" (Core Extraction) represents the integration of **Causal Graphs**. The model reads a `causal_adj_matrix.npy` (generated from tools like PCMCI or causal discovery) to inform the AD layers about which sensors *should* influence each other, making the attention mechanism physically grounded rather than purely black-box.
