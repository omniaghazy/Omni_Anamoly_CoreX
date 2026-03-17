"""
============================================================================
STANDALONE OmniAnomaly Kaggle Notebook
============================================================================
Paper : "Robust Anomaly Detection for Multivariate Time Series through
         Stochastic Recurrent Neural Network" (Su et al., KDD 2019)

Purpose: Self-contained reimplementation of the core OmniAnomaly math
         using ONLY TensorFlow 1.x ops (tf.compat.v1 / tf.Session).
         No zhusuan, no tfsnippet, no project imports.

         Designed to run on Kaggle (TF 2.x with v1 compat) and to match
         the exact tensor conventions of the Omni_CoreX_v2 project:
           - Input  : [batch_size, window_length, x_dim]
           - GRU    : Bidirectional, 2-layer stacked, 256 hidden
           - VAE    : z_dim=64, planar NF (20 layers)
           - Score  : per-sensor reconstruction log probability

Author : Auto-generated for Omnia's 15-sensor robotic-arm project
============================================================================
"""

# ── 0. Environment Setup ────────────────────────────────────────────────────
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
np.random.seed(42)

import tensorflow as tf

# Force TF 1.x behaviour on Kaggle (TF 2.x)
if hasattr(tf, 'compat'):
    tf = tf.compat.v1
    tf.disable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)

print(f"[✓] TensorFlow version : {tf.__version__ if hasattr(tf,'__version__') else 'v1-compat'}")
print(f"[✓] NumPy version      : {np.__version__}")

# ── 1. Hyperparameters (matching Omni_CoreX_v2/main.py ExpConfig) ────────────
NUM_SENSORS      = 15         # x_dim (robotic arm sensors)
WINDOW_LENGTH    = 120        # sliding window
Z_DIM            = 64         # latent dimension
RNN_HIDDEN       = 256        # GRU hidden size per direction
DENSE_DIM        = 256        # dense layer width
NF_LAYERS        = 20         # planar normalizing flow layers
BETA             = 0.5        # β-VAE weight
L2_REG           = 0.0001     # L2 regularisation
STD_EPSILON      = 1e-4       # softplus floor
BATCH_SIZE       = 64
MAX_EPOCH        = 5          # short run for Kaggle test
LEARNING_RATE    = 0.001
N_Z_SAMPLES      = 5          # Monte-Carlo z samples (importance sampling)

# Dummy data sizes
N_TRAIN          = 5000       # time steps for training
N_TEST           = 2000       # time steps for testing

print("\n[Config]")
print(f"  sensors={NUM_SENSORS}  window={WINDOW_LENGTH}  z_dim={Z_DIM}")
print(f"  rnn_hidden={RNN_HIDDEN}  nf_layers={NF_LAYERS}  beta={BETA}")
print(f"  batch_size={BATCH_SIZE}  max_epoch={MAX_EPOCH}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DUMMY DATA GENERATION (mimicking matlab_v1.csv / ready_for_ai.csv)
# ═══════════════════════════════════════════════════════════════════════════════
def generate_dummy_sensor_data(n_timesteps, n_sensors, anomaly_ratio=0.05):
    """
    Generate MinMax-scaled multivariate time series with injected anomalies.
    Mimics the 15-sensor robotic-arm data from the Omni_CoreX_v2 pipeline.

    Returns:
        data   : float32 array [n_timesteps, n_sensors] in [0, 1]
        labels : int array     [n_timesteps]  (1 = anomaly, 0 = normal)
    """
    t = np.linspace(0, 4 * np.pi, n_timesteps)
    data = np.zeros((n_timesteps, n_sensors), dtype=np.float32)

    for s in range(n_sensors):
        freq  = 0.5 + 0.3 * s
        phase = s * 0.4
        data[:, s] = (np.sin(freq * t + phase)
                      + 0.15 * np.random.randn(n_timesteps))

    # MinMax scale each sensor to [0, 1]
    mins = data.min(axis=0, keepdims=True)
    maxs = data.max(axis=0, keepdims=True)
    data = (data - mins) / (maxs - mins + 1e-8)

    # Inject point anomalies
    labels = np.zeros(n_timesteps, dtype=np.float32)
    n_anomalies = int(n_timesteps * anomaly_ratio)
    anom_idx = np.random.choice(n_timesteps, n_anomalies, replace=False)
    for idx in anom_idx:
        sensor = np.random.randint(0, n_sensors)
        data[idx, sensor] += np.random.uniform(0.5, 1.5) * np.random.choice([-1, 1])
        labels[idx] = 1.0

    return data.astype(np.float32), labels.astype(np.float32)


print("\n[Step 1] Generating dummy 15-sensor data...")
x_train_raw, _ = generate_dummy_sensor_data(N_TRAIN, NUM_SENSORS, anomaly_ratio=0.0)
x_test_raw, y_test = generate_dummy_sensor_data(N_TEST, NUM_SENSORS, anomaly_ratio=0.05)
print(f"  Train: {x_train_raw.shape}   Test: {x_test_raw.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SLIDING WINDOW (mirrors BatchSlidingWindow from utils.py)
# ═══════════════════════════════════════════════════════════════════════════════
def create_sliding_windows(data, window_size):
    """
    Convert [T, D] → [N_windows, window_size, D].
    Each window ends at index i: data[i-window_size+1 : i+1].
    """
    n = len(data) - window_size + 1
    windows = np.array([data[i:i + window_size] for i in range(n)],
                       dtype=np.float32)
    return windows


print("[Step 2] Creating sliding windows...")
x_train_win = create_sliding_windows(x_train_raw, WINDOW_LENGTH)
x_test_win  = create_sliding_windows(x_test_raw, WINDOW_LENGTH)
y_test_win  = y_test[WINDOW_LENGTH - 1:]  # align labels to last point of window
print(f"  Train windows: {x_train_win.shape}   Test windows: {x_test_win.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. HELPER FUNCTIONS (matching wrapper.py from Omni_CoreX_v2)
# ═══════════════════════════════════════════════════════════════════════════════
def softplus_std(x, units, epsilon, name):
    """
    Produce a strictly positive std via softplus + epsilon floor.
    Mirrors omni_anomaly/wrapper.py::softplus_std.
    """
    with tf.variable_scope(name):
        raw = tf.layers.dense(x, units, name='dense')
        std = tf.nn.softplus(raw)
        std = tf.maximum(std, epsilon) + 1e-8
    return std


def bidirectional_gru_encoder(x, rnn_hidden, dense_dim, name):
    """
    2-layer stacked bidirectional GRU → 2 dense layers → linear projection.
    Mirrors omni_anomaly/wrapper.py::rnn().

    Input : [batch, window_length, features]
    Output: [batch, window_length, rnn_hidden]
    """
    with tf.variable_scope(name):
        # Using tf.keras.layers for Keras 3 compatibility on Kaggle
        gru_fw = tf.keras.layers.RNN([tf.keras.layers.GRUCell(rnn_hidden) for _ in range(2)], return_sequences=True)
        gru_bw = tf.keras.layers.RNN([tf.keras.layers.GRUCell(rnn_hidden) for _ in range(2)], return_sequences=True, go_backwards=True)
        bidirectional_gru = tf.keras.layers.Bidirectional(layer=gru_fw, backward_layer=gru_bw)
        
        outputs = bidirectional_gru(x)
        # outputs: [batch, window, 2*rnn_hidden]

        # Self-attention (matching project's attention mechanism)
        att_score  = tf.layers.dense(outputs, 1, activation=tf.nn.tanh)
        att_weight = tf.nn.softmax(att_score, axis=1)
        outputs    = outputs * att_weight

        # Dense layers
        for i in range(2):
            outputs = tf.layers.dense(outputs, dense_dim,
                                      activation=tf.nn.relu,
                                      name=f'dense_{i}')
        # Final projection to rnn_hidden
        outputs = tf.layers.dense(outputs, rnn_hidden, name='proj')
    return outputs


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PLANAR NORMALIZING FLOW
# ═══════════════════════════════════════════════════════════════════════════════
class PlanarNormalizingFlow:
    """
    Implements a chain of planar normalizing flow transformations.
    Each layer:  z' = z + u * tanh(w^T z + b)
    log_det_jac = log|1 + u^T * dtanh * w|

    Reference: Rezende & Mohamed (2015), same as tfsnippet's
    planar_normalizing_flows used in the project.
    """

    def __init__(self, z_dim, n_layers, name='planar_nf'):
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.name = name
        self._params = []

    def build(self):
        """Create TF variables for all NF layers."""
        with tf.variable_scope(self.name):
            for i in range(self.n_layers):
                with tf.variable_scope(f'layer_{i}'):
                    w = tf.get_variable('w', [1, self.z_dim],
                                        initializer=tf.glorot_uniform_initializer())
                    u = tf.get_variable('u', [1, self.z_dim],
                                        initializer=tf.glorot_uniform_initializer())
                    b = tf.get_variable('b', [1],
                                        initializer=tf.zeros_initializer())
                    self._params.append((w, u, b))

    def transform(self, z):
        """
        Apply the NF chain.

        Args:
            z: [*, z_dim]  (supports arbitrary leading dims)

        Returns:
            z_transformed: same shape as z
            sum_log_det  : [*]  (summed log determinants)
        """
        sum_log_det = tf.zeros(tf.shape(z)[:-1])
        for w, u_raw, b in self._params:
            # Enforce invertibility: u_hat from Appendix A of Rezende (2015)
            wtu = tf.reduce_sum(w * u_raw, axis=-1, keepdims=True)  # [1, 1]
            m_wtu = -1.0 + tf.nn.softplus(wtu)
            u = u_raw + (m_wtu - wtu) * w / (tf.reduce_sum(w * w, axis=-1,
                                                             keepdims=True) + 1e-8)

            # z' = z + u * tanh(w^T z + b)
            wtz = tf.reduce_sum(z * w, axis=-1, keepdims=True) + b  # [*, 1]
            tanh_wtz = tf.nn.tanh(wtz)                              # [*, 1]
            z = z + u * tanh_wtz                                     # [*, z_dim]

            # log|det J| = log|1 + u^T * (1 - tanh^2) * w|
            dtanh = 1.0 - tanh_wtz ** 2                              # [*, 1]
            utw   = tf.reduce_sum(u * w, axis=-1, keepdims=True)     # [1, 1]
            log_det = tf.log(tf.abs(1.0 + dtanh * utw) + 1e-8)      # [*, 1]
            sum_log_det += tf.squeeze(log_det, axis=-1)

        return z, sum_log_det


# ═══════════════════════════════════════════════════════════════════════════════
# 6. STOCHASTIC VARIABLE CONNECTION (RecurrentDistribution logic)
# ═══════════════════════════════════════════════════════════════════════════════
def stochastic_variable_connection(input_q_time_first, z_dim, n_samples, name):
    """
    Implements the recurrent latent sampling from Section 3.2 of the paper:
        z_t ~ N(mu_t, sigma_t)
        [mu_t, sigma_t] = f(h_t, z_{t-1})

    Uses tf.scan over time steps, matching RecurrentDistribution in the project.

    Args:
        input_q_time_first: [window_length, batch, rnn_hidden]  (time-first GRU output)
        z_dim:  latent dimension
        n_samples: number of z samples (importance sampling)
        name: variable scope

    Returns:
        z_samples: [n_samples, batch, window_length, z_dim]
        mu_all   : [n_samples, batch, window_length, z_dim]
        std_all  : [n_samples, batch, window_length, z_dim]
    """
    with tf.variable_scope(name):
        batch_size = tf.shape(input_q_time_first)[1]
        hidden_dim = input_q_time_first.get_shape().as_list()[-1]

        # Generate noise for reparameterisation trick
        # Shape: [window_length, n_samples, batch, z_dim]
        noise = tf.random_normal(
            [WINDOW_LENGTH, n_samples, batch_size, z_dim])

        # Scan function: processes one time step
        def scan_step(state, inputs):
            z_prev, _, _ = state
            noise_t, h_t = inputs

            # Expand h_t for n_samples: [batch, hidden] → [n_samples, batch, hidden]
            h_t_exp = tf.tile(tf.expand_dims(h_t, 0), [n_samples, 1, 1])

            # Concatenate previous z with current hidden state
            combined = tf.concat([h_t_exp, z_prev], axis=-1)
            # combined: [n_samples, batch, hidden + z_dim]

            # Parameters network
            mu_t = tf.layers.dense(combined, z_dim, name='mu',
                                   reuse=tf.AUTO_REUSE)
            std_t = softplus_std(combined, z_dim, STD_EPSILON, name='std')

            # Reparameterisation trick: z_t = mu + std * epsilon
            z_t = mu_t + std_t * noise_t

            return z_t, mu_t, std_t

        # Initial state
        init_z   = tf.zeros([n_samples, batch_size, z_dim])
        init_mu  = tf.zeros([n_samples, batch_size, z_dim])
        init_std = tf.ones([n_samples, batch_size, z_dim])

        # Temporal scan
        z_all, mu_all, std_all = tf.scan(
            scan_step,
            elems=(noise, input_q_time_first),
            initializer=(init_z, init_mu, init_std),
            name='stochastic_scan'
        )
        # z_all: [window_length, n_samples, batch, z_dim]

        # Transpose to [n_samples, batch, window_length, z_dim]
        z_all   = tf.transpose(z_all,   [1, 2, 0, 3])
        mu_all  = tf.transpose(mu_all,  [1, 2, 0, 3])
        std_all = tf.transpose(std_all, [1, 2, 0, 3])

    return z_all, mu_all, std_all


# ═══════════════════════════════════════════════════════════════════════════════
# 7. BUILD THE FULL COMPUTATION GRAPH
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[Step 3] Building TF computation graph...")

tf.reset_default_graph()

# ── 7.1 Input placeholder ──────────────────────────────────────────────────
input_x = tf.placeholder(tf.float32,
                          shape=[None, WINDOW_LENGTH, NUM_SENSORS],
                          name='input_x')

# ── 7.2 Encoder: GRU → hidden representation ───────────────────────────────
with tf.variable_scope('encoder'):
    h_x = bidirectional_gru_encoder(input_x, RNN_HIDDEN, DENSE_DIM,
                                    name='gru_encoder')
    # h_x: [batch, window_length, RNN_HIDDEN]

    # Transpose to time-first for tf.scan
    h_x_time_first = tf.transpose(h_x, [1, 0, 2])
    # h_x_time_first: [window_length, batch, RNN_HIDDEN]

# ── 7.3 Stochastic Variable Connection → z samples ─────────────────────────
z_samples, z_mu, z_std = stochastic_variable_connection(
    h_x_time_first, Z_DIM, N_Z_SAMPLES, name='svc')
# z_samples: [n_samples, batch, window_length, z_dim]

# ── 7.4 Planar Normalizing Flow on z ───────────────────────────────────────
nf = PlanarNormalizingFlow(Z_DIM, NF_LAYERS, name='nf')
nf.build()

# Flatten leading dims for NF, then reshape back
z_flat_shape = tf.shape(z_samples)
z_flat = tf.reshape(z_samples, [-1, Z_DIM])
z_nf, nf_log_det = nf.transform(z_flat)
z_nf = tf.reshape(z_nf, z_flat_shape)
nf_log_det = tf.reshape(nf_log_det, z_flat_shape[:-1])
# z_nf: [n_samples, batch, window_length, z_dim]

# ── 7.5 Decoder: z → reconstructed x ───────────────────────────────────────
with tf.variable_scope('decoder'):
    # Average z over samples for decoder input
    z_mean_sample = tf.reduce_mean(z_nf, axis=0)
    # z_mean_sample: [batch, window_length, z_dim]

    h_z = bidirectional_gru_encoder(z_mean_sample, RNN_HIDDEN, DENSE_DIM,
                                    name='gru_decoder')
    # h_z: [batch, window_length, RNN_HIDDEN]

    x_mu  = tf.layers.dense(h_z, NUM_SENSORS, name='x_mu')
    x_std = softplus_std(h_z, NUM_SENSORS, STD_EPSILON, name='x_std')
    # x_mu, x_std: [batch, window_length, NUM_SENSORS]

# ── 7.6 Reconstruction log probability (anomaly score) ─────────────────────
with tf.name_scope('anomaly_score'):
    # Gaussian log prob per sensor (group_ndims=0 matches project)
    # log N(x | mu, std) = -0.5*log(2π) - log(std) - 0.5*((x-mu)/std)^2
    log_2pi = tf.constant(0.5 * np.log(2.0 * np.pi), dtype=tf.float32)
    sq_diff = tf.square((input_x - x_mu) / (x_std + 1e-8))
    recon_log_prob = -log_2pi - tf.log(x_std + 1e-8) - 0.5 * sq_diff
    # recon_log_prob: [batch, window_length, NUM_SENSORS]

    # Anomaly score = negative log prob of LAST time step (real-time detection)
    score_last_point = -recon_log_prob[:, -1, :]
    # score_last_point: [batch, NUM_SENSORS]

# ── 7.7 Training loss: β-VAE ELBO ──────────────────────────────────────────
with tf.name_scope('training_loss'):
    # Reconstruction term: sum log p(x|z) over time and sensors
    recon_loss = -tf.reduce_sum(recon_log_prob, axis=[1, 2])
    # recon_loss: [batch]

    # KL divergence: KL(q(z|x) || N(0,1))
    # = 0.5 * sum(mu^2 + std^2 - log(std^2) - 1)
    kl_div = 0.5 * tf.reduce_sum(
        tf.square(z_mu) + tf.square(z_std) - 2.0 * tf.log(z_std + 1e-8) - 1.0,
        axis=-1)
    # kl_div: [n_samples, batch, window_length]
    kl_div = tf.reduce_mean(kl_div, axis=0)  # avg over samples
    kl_div = tf.reduce_sum(kl_div, axis=1)   # sum over time
    # kl_div: [batch]

    # NF correction: subtract log det jacobian
    nf_correction = tf.reduce_mean(
        tf.reduce_sum(nf_log_det, axis=-1), axis=0)
    # nf_correction: [batch]

    # β-VAE loss
    elbo = recon_loss + BETA * kl_div - nf_correction
    loss = tf.reduce_mean(elbo)

    # L2 regularisation
    trainable = tf.trainable_variables()
    l2_loss = tf.add_n([tf.nn.l2_loss(v)
                        for v in trainable if 'bias' not in v.name])
    total_loss = loss + L2_REG * l2_loss

# ── 7.8 Optimizer ───────────────────────────────────────────────────────────
optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
grads_and_vars = optimizer.compute_gradients(total_loss)
clipped = [(tf.clip_by_norm(g, 5.0), v) for g, v in grads_and_vars if g is not None]
train_op = optimizer.apply_gradients(clipped)

print("[✓] Graph built successfully.")
print(f"    Trainable parameters: {sum(int(np.prod(v.shape)) for v in trainable):,}")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[Step 4] Training...")

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    n_train_windows = len(x_train_win)

    for epoch in range(1, MAX_EPOCH + 1):
        # Shuffle training data each epoch
        perm = np.random.permutation(n_train_windows)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train_windows, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_train_windows)
            batch_idx = perm[start:end]
            batch_x = x_train_win[batch_idx]

            _, batch_loss = sess.run(
                [train_op, total_loss],
                feed_dict={input_x: batch_x})
            epoch_loss += batch_loss
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  Epoch {epoch}/{MAX_EPOCH}  |  Loss: {avg_loss:.4f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # 9. ANOMALY SCORING (forward pass on test set)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[Step 5] Computing anomaly scores on test set...")

    all_scores = []
    for start in range(0, len(x_test_win), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(x_test_win))
        batch_x = x_test_win[start:end]
        batch_scores = sess.run(score_last_point,
                                feed_dict={input_x: batch_x})
        all_scores.append(batch_scores)

    anomaly_scores = np.concatenate(all_scores, axis=0)
    # anomaly_scores: [N_test_windows, NUM_SENSORS]

print("\n" + "=" * 60)
print("          OmniAnomaly Standalone — Results")
print("=" * 60)
print(f"Anomaly scores shape : {anomaly_scores.shape}")
print(f"Score range          : [{anomaly_scores.min():.4f}, {anomaly_scores.max():.4f}]")
print(f"Score mean           : {anomaly_scores.mean():.4f}")
print(f"Score std            : {anomaly_scores.std():.4f}")

# Per-sensor aggregated score (mean over test windows)
per_sensor_score = anomaly_scores.mean(axis=0)
print(f"\nPer-sensor mean anomaly scores (15 sensors):")
for i, s in enumerate(per_sensor_score):
    bar = "█" * int(min(s / per_sensor_score.max() * 30, 30))
    print(f"  Sensor {i:2d}: {s:8.4f}  {bar}")

# Aggregate to single score per window (sum over sensors)
agg_scores = anomaly_scores.sum(axis=1)
print(f"\nAggregated score shape : {agg_scores.shape}")
print(f"Aggregated range       : [{agg_scores.min():.4f}, {agg_scores.max():.4f}]")

# Simple threshold-based detection (top 5% as anomalies)
threshold = np.percentile(agg_scores, 95)
predicted = (agg_scores > threshold).astype(np.float32)
y_aligned = y_test_win[:len(predicted)]

tp = np.sum((predicted == 1) & (y_aligned == 1))
fp = np.sum((predicted == 1) & (y_aligned == 0))
fn = np.sum((predicted == 0) & (y_aligned == 1))
prec = tp / (tp + fp + 1e-8)
rec  = tp / (tp + fn + 1e-8)
f1   = 2 * prec * rec / (prec + rec + 1e-8)

print(f"\n[Quick Eval] Threshold (95th pct): {threshold:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1-Score  : {f1:.4f}")
print("=" * 60)
print("\n[✓] PHASE 2 COMPLETE. Standalone notebook executed successfully.")
print("[!] PHASE 3: HARD STOP — No integration code follows.")
