# Walkthrough: Fixing the Scoring Phase Crash (ValueError)

This document details the resolution of the `ValueError` that occurred during the transition from the Training phase to the Scoring/Evaluation phase in OmniAnomaly CoreX V2.

## 1. The Error Message

During the scoring phase (`predictor.get_score`), the model would crash with the following error:

```text
ValueError: Variable model/model/p_x_given_z/x_mean/kernel already exists, disallowed. 
Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope?
```

## 2. Root Cause Analysis

In TensorFlow 1.15, variables created via `tf.layers.dense` (used inside `mean_layer` and `std_layer`) are assigned to a specific variable scope path.

1.  **Training Phase**: The `Trainer` builds the graph and creates the variables (e.g., `p_x_given_z/x_mean/kernel`) for the first time.
2.  **Scoring Phase**: The `Predictor` (in `main.py`) calls `self._build_graph()`, which attempts to re-run the same model logic to compute results for the test set.
3.  **The Conflict**: Because the `mean_layer` and `std_layer` calls were physically outside any scope that allowed reuse, TensorFlow's default behavior was to attempt *creating* new variables. Since they already existed from the training phase, it threw a `ValueError`.

## 3. The Implementation (The Fix)

Simply indenting the layers into a new scope like `with tf.variable_scope('hidden'):` would have changed the variable name (e.g., to `p_x_given_z/hidden/x_mean`), which would make them incompatible with the trained weights.

### The Solution:
We wrapped the layer executions in `wrapper.py` using `tf.AUTO_REUSE` while explicitly passing the current active scope. This allows the layers to **reuse** the existing trained weights without changing their hierarchical names.

**File: `omni_anomaly/wrapper.py`**

```python
# Before Fix:
z_mean = mean_layer(h)
z_std = softplus_std(h, ...)

# After Fix:
with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    z_mean = mean_layer(h)
    z_std = softplus_std(h, units=z_mean.get_shape().as_list()[-1], name='z_std_secure')
```

## 4. Why this works
- `tf.get_variable_scope()`: Returns the current scope (`model/p_x_given_z`).
- `reuse=tf.AUTO_REUSE`: Tells TensorFlow: "If the variables already exist, use them. If they don't, create them."

This ensures that the Scoring phase uses the **exact same** weights that were just optimized during the Training phase, allowing the model to finish the full 1-epoch run successfully.
