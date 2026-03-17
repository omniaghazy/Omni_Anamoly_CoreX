"""
============================================================================
STANDALONE AD-VAE Kaggle Notebook
============================================================================
Paper : "An Anomaly Detection Method for Multivariate Time Series Data
         Based on Variational Autoencoders and Association Discrepancy"
         (Wang & Zhang, 2025)

Constraints Strictly Followed:
 - TF 1.x ONLY (tf.compat.v1, tf.Session, tf.layers)
 - Designed to run on Kaggle
 - Data shape: [batch_size, window_length, 15]
 - No external project dependencies

Author: Antigravity for Omnia_CoreX Phase 2
============================================================================
"""

import os
import numpy as np
import tensorflow as tf
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if hasattr(tf, 'compat'):
    tf = tf.compat.v1
    tf.disable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)

print(f"[✓] TF 1.x Compatibility Mode Enabled. Version: {tf.__version__}")

# --- CONFIGURATION ---
DATASET_MODE = "DUMMY"
CUSTOM_CSV_PATH = "/kaggle/input/robot-arm-ready/ready_for_ai.csv"
X_DIM = 15
WINDOW_LENGTH = 100
D_MODEL = 512
N_HEADS = 8
ADM_LAYERS = 3
LATENT_DIM = 64
K_WEIGHT = 3.0
BATCH_SIZE = 32
MAX_EPOCH = 5

def get_data():
    if DATASET_MODE == "DUMMY":
        t = np.linspace(0, 100, 5000)
        data = np.array([np.sin(t * (0.1 + i*0.05)) + np.random.normal(0, 0.05, 5000) for i in range(15)]).T
        return data.astype(np.float32), 15
    else:
        import pandas as pd
        df = pd.read_csv(CUSTOM_CSV_PATH)
        data = df.select_dtypes(include=[np.number]).fillna(method='ffill').fillna(0).values[:, :X_DIM]
        from sklearn.preprocessing import MinMaxScaler
        data = MinMaxScaler().fit_transform(data)
        return data.astype(np.float32), X_DIM

data_raw, _ = get_data()

# --- MATH CORE: ASSOCIATION DISCREPANCY ---

def positional_encoding(window_length, d_model):
    position = np.arange(window_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe = np.zeros((window_length, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return tf.constant(pe, dtype=tf.float32)

def kl_divergence_matrix(p, q):
    # p, q shape: [batch, heads, window, window]
    return tf.reduce_sum(p * tf.log(p / (q + 1e-8) + 1e-8), axis=-1)

def association_discrepancy_layer(x, d_model, n_heads, name, reuse=False):
    """
    Computes Prior Association (P) and Sequence Association (S).
    """
    batch_size = tf.shape(x)[0]
    window_len = tf.shape(x)[1]
    depth = d_model // n_heads

    with tf.variable_scope(name, reuse=reuse):
        Q = tf.layers.dense(x, d_model, name="W_Q")
        K = tf.layers.dense(x, d_model, name="W_K")
        V = tf.layers.dense(x, d_model, name="W_V")
        sigma_raw = tf.layers.dense(x, n_heads, activation=tf.nn.softplus, name="W_sigma") # [batch, window, heads]

        Q = tf.reshape(Q, [batch_size, window_len, n_heads, depth])
        K = tf.reshape(K, [batch_size, window_len, n_heads, depth])
        V = tf.reshape(V, [batch_size, window_len, n_heads, depth])

        Q = tf.transpose(Q, [0, 2, 1, 3]) # [batch, heads, window, depth]
        K = tf.transpose(K, [0, 2, 1, 3])
        V = tf.transpose(V, [0, 2, 1, 3])

        # 1. Sequence Association (S) via Softmax Attention
        scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(tf.cast(depth, tf.float32))
        S = tf.nn.softmax(scores, axis=-1)

        # 2. Prior Association (P) via Learned Gaussian Distance
        sigma = tf.transpose(sigma_raw, [0, 2, 1]) + 1e-6 # [batch, heads, window]
        idx = tf.cast(tf.range(window_len), tf.float32)
        dist = tf.expand_dims(idx, 0) - tf.expand_dims(idx, 1) # [window, window]
        dist_sq = tf.square(dist) # [window, window]
        
        # Expand for calculation: [batch, heads, window, window]
        dist_sq_exp = tf.expand_dims(tf.expand_dims(dist_sq, 0), 0)
        sigma_exp = tf.expand_dims(sigma, -1)
        
        P_unnorm = tf.exp(-dist_sq_exp / (2.0 * tf.square(sigma_exp) + 1e-8)) / (tf.sqrt(2 * np.pi) * sigma_exp + 1e-8)
        P = P_unnorm / (tf.reduce_sum(P_unnorm, axis=-1, keepdims=True) + 1e-8)

        # Output Features e_l
        attn_out = tf.matmul(S, V)
        attn_out = tf.transpose(attn_out, [0, 2, 1, 3])
        attn_out = tf.reshape(attn_out, [batch_size, window_len, d_model])
        
        # Residual
        x_out = tf.contrib.layers.layer_norm(attn_out + x)

        # Compute Layer Association Discrepancy: AssDis
        AssDis = kl_divergence_matrix(P, S) + kl_divergence_matrix(S, P)
        AssDis = tf.reduce_mean(AssDis, axis=1) # Average over heads -> [batch, window]

        return x_out, P, S, AssDis

def build_model(input_x):
    # Base Embedding + Positional Encoding
    emb = tf.layers.dense(input_x, D_MODEL, name="input_emb")
    pe = positional_encoding(WINDOW_LENGTH, D_MODEL)
    x_enc = tf.contrib.layers.layer_norm(emb + tf.expand_dims(pe, 0))

    P_list, S_list, AssDis_list = [], [], []
    
    # 1. Association Discrepancy Layer (Encoder)
    for l in range(ADM_LAYERS):
        x_enc, P_l, S_l, AssDis_l = association_discrepancy_layer(x_enc, D_MODEL, N_HEADS, name=f"ADM_{l}")
        P_list.append(P_l)
        S_list.append(S_l)
        AssDis_list.append(AssDis_l)

    # 2. VAE Reconstruction Layer (Latent Bottleck)
    with tf.variable_scope("vae"):
        h_flat = tf.reshape(tf.layers.dense(x_enc, D_MODEL), [-1, WINDOW_LENGTH * D_MODEL])
        z_mu = tf.layers.dense(h_flat, LATENT_DIM)
        z_logvar = tf.layers.dense(h_flat, LATENT_DIM)
        
        # Reparameterization
        z_std = tf.exp(0.5 * z_logvar)
        z = z_mu + z_std * tf.random_normal(tf.shape(z_mu))
        
        h_dec = tf.layers.dense(z, WINDOW_LENGTH * D_MODEL, activation=tf.nn.relu)
        h_dec_seq = tf.reshape(h_dec, [-1, WINDOW_LENGTH, D_MODEL])
        
        # Reconstruct X
        x_hat = tf.layers.dense(h_dec_seq, X_DIM, name="x_reconstruct")
        
    # 3. Stochastic Association Discrepancy Layer (From Reconstructed X_hat)
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        emb_hat = tf.layers.dense(x_hat, D_MODEL, name="input_emb")
        x_enc_hat = tf.contrib.layers.layer_norm(emb_hat + tf.expand_dims(pe, 0))
        
        P_hat_list, S_hat_list, AssDis_hat_list = [], [], []
        for l in range(ADM_LAYERS):
             x_enc_hat, P_l_hat, S_l_hat, AssDis_l_hat = association_discrepancy_layer(x_enc_hat, D_MODEL, N_HEADS, name=f"ADM_{l}", reuse=True)
             P_hat_list.append(P_l_hat)
             S_hat_list.append(S_l_hat)
             AssDis_hat_list.append(AssDis_l_hat)

    # Aggregate Discrepancy across layers
    total_AssDis = tf.reduce_mean(tf.stack(AssDis_list, axis=0), axis=0) # [batch, window]
    total_AssDis_hat = tf.reduce_mean(tf.stack(AssDis_hat_list, axis=0), axis=0)

    # Loss Calculation
    MSE = tf.reduce_mean(tf.square(input_x - x_hat), axis=-1) # Pointwise [batch, window]
    KLD = -0.5 * tf.reduce_sum(1 + z_logvar - tf.square(z_mu) - tf.exp(z_logvar), axis=1) # [batch]
    KLD_expanded = tf.expand_dims(KLD, 1) / float(WINDOW_LENGTH)

    # Min-Max Discrepancy Loss
    loss_AssDis_min = 0.0
    loss_AssDis_max = 0.0
    for Pl, Sl, Pl_hat, Sl_hat in zip(P_list, S_list, P_hat_list, S_hat_list):
        # Min Phase (P tracks S_detach)
        S_detach, S_hat_detach = tf.stop_gradient(Sl), tf.stop_gradient(Sl_hat)
        loss_AssDis_min += tf.reduce_mean(kl_divergence_matrix(Pl, S_detach) + kl_divergence_matrix(S_detach, Pl))
        loss_AssDis_min += tf.reduce_mean(kl_divergence_matrix(Pl_hat, S_hat_detach) + kl_divergence_matrix(S_hat_detach, Pl_hat))
        
        # Max Phase (S moves away from P_detach)
        P_detach, P_hat_detach = tf.stop_gradient(Pl), tf.stop_gradient(Pl_hat)
        loss_AssDis_max -= tf.reduce_mean(kl_divergence_matrix(P_detach, Sl) + kl_divergence_matrix(Sl, P_detach))
        loss_AssDis_max -= tf.reduce_mean(kl_divergence_matrix(P_hat_detach, Sl_hat) + kl_divergence_matrix(Sl_hat, P_hat_detach))
        
    avg_AssDis_min = loss_AssDis_min / (2.0 * ADM_LAYERS)
    avg_AssDis_max = loss_AssDis_max / (2.0 * ADM_LAYERS)

    MSE_total = tf.reduce_mean(MSE)
    KLD_total = tf.reduce_mean(KLD)

    # Total Joint Optimization Objectives
    loss_min = MSE_total + KLD_total + K_WEIGHT * avg_AssDis_min
    loss_max = MSE_total + KLD_total + K_WEIGHT * avg_AssDis_max

    # 4. Final Anomaly Score (Eq. 16 in paper)
    # Score = Softmax(-(AssDis(X) + AssDis(X_hat))) * (MSE + KLD)
    combined_discrepancy = total_AssDis + total_AssDis_hat # [batch, window]
    discrepancy_weight = tf.nn.softmax(-combined_discrepancy, axis=1) # [batch, window]
    anomaly_score = discrepancy_weight * (MSE + KLD_expanded) # [batch, window]
    
    return loss_min, loss_max, anomaly_score, x_hat


tf.reset_default_graph()
input_x = tf.placeholder(tf.float32, [None, WINDOW_LENGTH, X_DIM])

loss_min, loss_max, anomaly_score, x_hat = build_model(input_x)

optimizer = tf.train.AdamOptimizer(1e-4)

# We define two train ops to alternate min/max phases
train_op_min = optimizer.minimize(loss_min)
train_op_max = optimizer.minimize(loss_max)

print("[✓] AD-VAE Computational Graph Successfully Constructed.")

# --- EXECUTION ---
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Dummy Sliding Windows
windows = np.array([data_raw[i:i+WINDOW_LENGTH] for i in range(len(data_raw) - WINDOW_LENGTH)])

print("\n[!] Starting Optimization (Min-Max Alternation)")
for epoch in range(1, MAX_EPOCH + 1):
    indices = np.random.choice(len(windows), BATCH_SIZE)
    batch = windows[indices]
    
    # Alternating optimization
    _, l_min = sess.run([train_op_min, loss_min], {input_x: batch})
    _, l_max = sess.run([train_op_max, loss_max], {input_x: batch})
    
    print(f"  Epoch {epoch}/{MAX_EPOCH} | Loss Min Phase: {l_min:.4f} | Loss Max Phase: {l_max:.4f}")

scores = sess.run(anomaly_score, {input_x: windows[:5]})
print("\n[✓] STANDALONE EVALUATION COMPLETE.")
print(f"Anomaly Score Output Shape: {scores.shape} -> Expected [batch_size, window_length]")
