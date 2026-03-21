# -*- coding: utf-8 -*-
from functools import partial
import os

import numpy as np
import tensorflow.compat.v1 as tf
import tfsnippet as spt
from tensorflow.python.ops.linalg.linear_operator_identity import LinearOperatorIdentity
from tensorflow_probability.python.distributions import LinearGaussianStateSpaceModel, MultivariateNormalDiag
from tfsnippet.distributions import Normal
from tfsnippet.utils import VarScopeObject, reopen_variable_scope
from tfsnippet.variational import VariationalInference

from omni_anomaly.recurrent_distribution import RecurrentDistribution
from omni_anomaly.vae import Lambda, VAE
from omni_anomaly.wrapper import TfpDistribution, softplus_std, rnn, wrap_params_net


# =============================================================================
# CausalGraphModule  (TensorFlow 1.x equivalent of the PyTorch version from
# the entropy-graph notebook, based on the CGAD paper)
#
# This module implements a 2-layer Graph Convolutional Network (GCN) that uses
# a pre-computed Transfer Entropy adjacency matrix to propagate causal
# information across the 15 robotic-arm sensors.
# =============================================================================
class CausalGraphModule(object):
    """
    A 2-layer GCN with a residual connection, operating on a fixed causal
    adjacency matrix derived from Transfer Entropy.

    Expected input shape : [batch_size, num_sensors, in_channels]
    Output shape         : [batch_size, num_sensors, out_channels]
                           (out_channels must equal in_channels for the residual)
    """

    def __init__(self, adj_matrix, in_channels, out_channels, name='causal_graph'):
        """
        Args:
            adj_matrix : numpy ndarray of shape [num_sensors, num_sensors].
                         The Transfer Entropy causal adjacency matrix.
            in_channels  : int – feature dimension of the input  (last axis).
            out_channels : int – feature dimension of the output (last axis).
                           Must equal in_channels for the residual shortcut.
            name : str – TensorFlow variable scope name.
        """
        self._name = name
        self._in_channels = in_channels
        self._out_channels = out_channels

        # ── Normalize the adjacency matrix (D^{-1/2} A D^{-1/2}) ────────────
        # Same formula as the PyTorch version:
        #   deg = sum(A, axis=1)
        #   deg_inv_sqrt = (deg + eps)^{-0.5}
        #   norm_A = A * deg_inv_sqrt[:, None] * deg_inv_sqrt[None, :]
        A = adj_matrix.astype(np.float32)
        deg = A.sum(axis=1)
        deg_inv_sqrt = np.power(deg + 1e-5, -0.5)
        norm_A = A * deg_inv_sqrt[:, None] * deg_inv_sqrt[None, :]
        self._norm_A = norm_A                       # stored as numpy; converted
                                                    # to a TF constant on first call

    def __call__(self, x):
        """
        Forward pass of the 2-layer GCN with residual connection.

        Args:
            x : tf.Tensor of shape [batch_size, num_sensors (15), in_channels]

        Returns:
            tf.Tensor of shape [batch_size, num_sensors (15), out_channels]
        """
        with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
            # ── Constant adjacency (created once, reused) ────────────────────
            # norm_A shape: [num_sensors, num_sensors]  (e.g. [15, 15])
            norm_A = tf.constant(self._norm_A, dtype=tf.float32, name='norm_A')

            # ── Layer 0: Linear projection  →  ReLU ─────────────────────────
            # x  shape: [batch, 15, in_channels]
            # out shape: [batch, 15, 32]
            out = tf.layers.dense(x, 32, activation=tf.nn.relu, name='W0')
            # out shape: [batch, 15, 32]

            # ── Graph convolution: multiply by normalized adjacency ──────────
            # norm_A: [15, 15]   out: [batch, 15, 32]
            # Result: [batch, 15, 32]  –  each sensor aggregates its neighbours
            out = tf.einsum('ij,bjk->bik', norm_A, out)

            # ── Layer 1: Linear projection (no activation before residual) ───
            # W1 shape: [32, out_channels]
            out = tf.layers.dense(out, self._out_channels, name='W1')
            # out shape: [batch, 15, out_channels]

            # ── Residual connection ──────────────────────────────────────────
            # out + x (requires in_channels == out_channels)
            return out + x


# ─────────────────────────────────────────────────────────────────────────────
# [AD-VAE] Association Discrepancy Helpers (Wang & Zhang 2025)
# Optimized for TF 1.15 legacy tf.layers
# ─────────────────────────────────────────────────────────────────────────────
def kl_divergence_matrix(p, q):
    """Symmetric KL divergence for attention matrices."""
    return tf.reduce_sum(p * tf.log(p / (q + 1e-8) + 1e-8), axis=-1)

def association_layer(x, heads, d_model, name, window_len, reuse=False):
    """
    Computes Prior Association (P) and Sequence Association (S).
    Matches the Wang & Zhang 2025 mathematical framework.
    """
    with tf.variable_scope(name, reuse=reuse):
        # projections via tf.layers.dense (legacy)
        Q = tf.layers.dense(x, d_model, name="Q")
        K = tf.layers.dense(x, d_model, name="K")
        V = tf.layers.dense(x, d_model, name="V")
        sigma = tf.layers.dense(x, heads, activation=tf.nn.softplus, name="sigma")
        
        depth = d_model // heads
        def split_heads(t):
            t = tf.reshape(t, [-1, window_len, heads, depth])
            return tf.transpose(t, [0, 2, 1, 3])
            
        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # 1. Sequence Association (S) - Self Attention
        scores = tf.matmul(Q, K, transpose_b=True) / np.sqrt(depth)
        S = tf.nn.softmax(scores, axis=-1)

        # 2. Prior Association (P) - Learnable Gaussian Distance
        t_idx = tf.cast(tf.range(window_len), tf.float32)
        dist = tf.expand_dims(t_idx, 0) - tf.expand_dims(t_idx, 1)
        dist_sq = tf.expand_dims(tf.expand_dims(tf.square(dist), 0), 0)
        sigma_exp = tf.expand_dims(tf.transpose(sigma, [0, 2, 1]), -1)
        
        # P ~ N(0, sigma^2) based on temporal distance
        P = tf.exp(-dist_sq / (2.0 * tf.square(sigma_exp) + 1e-8)) / (np.sqrt(2.0*np.pi)*sigma_exp + 1e-8)
        P = P / (tf.reduce_sum(P, axis=-1, keepdims=True) + 1e-8)

        # Output + Residual
        out = tf.reshape(tf.transpose(tf.matmul(S, V), [0, 2, 1, 3]), [-1, window_len, d_model])
        # [FIX] Use tf.contrib.layers.layer_norm ONLY if available, otherwise fallback to simple norm
        try:
            from tensorflow.contrib import layers as contrib_layers
            res = contrib_layers.layer_norm(out + x)
        except (ImportError, AttributeError):
            # Simple fallback for environments where contrib is stripped
            epsilon = 1e-6
            mean, variance = tf.nn.moments(out + x, [-1], keep_dims=True)
            res = (out + x - mean) / tf.sqrt(variance + epsilon)
        
        # ─────────────────────────────────────────────────────────────────────
        # [AD-VAE] Min-Max Adversarial Strategy (Wang & Zhang 2025)
        # Using tf.stop_gradient to implement separate optimization goals:
        # P minimizes discrepancy (tracking S)
        # S maximizes discrepancy (diffentiating anomalies)
        # ─────────────────────────────────────────────────────────────────────
        # Discrepancy = Symmetric KL
        def sym_kl(p_mat, q_mat):
            return kl_divergence_matrix(p_mat, q_mat) + kl_divergence_matrix(q_mat, p_mat)
            
        dis_min = tf.reduce_mean(sym_kl(P, tf.stop_gradient(S)), axis=1) # For Prior
        dis_max = tf.reduce_mean(sym_kl(tf.stop_gradient(P), S), axis=1) # For Sequence
        
        return res, dis_min, dis_max, S, P


class OmniAnomaly(VarScopeObject):
    def __init__(self, config, name=None, scope=None):
            self._window_length = config.window_length
            self._x_dims = config.x_dim
            self.beta = getattr(config, 'beta', 1.0)
            self.config = config
            super(OmniAnomaly, self).__init__(name=name, scope=scope)
            
            # [AD-VAE] Track temporal discrepancies for Min-Max adversarial loss
            self._discrepancy_min_list = []
            self._discrepancy_max_list = []
            
            with reopen_variable_scope(self.variable_scope):
                # ═════════════════════════════════════════════════════════════
                # [CGAD] Load the pre-calculated Transfer Entropy causal
                # adjacency matrix and build the CausalGraphModule that will
                # process the raw sensor input *before* it enters the GRU.
                # ═════════════════════════════════════════════════════════════
                matrix_path = getattr(config, 'causal_adj_matrix_path',
                                      'causal_adj_matrix.npy')
                A_matrix = np.load(matrix_path)
                print(f'[CGAD] Loaded causal adjacency matrix from: {matrix_path}  '
                      f'shape={A_matrix.shape}')

                # CausalGraphModule for the raw input layer
                # in_channels=1 and out_channels=1 because each sensor is
                # treated as a single-feature node on the 15-node graph.
                self.causal_input = CausalGraphModule(
                    adj_matrix=A_matrix,
                    in_channels=1,
                    out_channels=1,
                    name='causal_input'
                )

                # 1. تعريف الـ Normalizing Flows لتحسين الـ Latent Space
                if config.posterior_flow_type == 'nf':
                    self._posterior_flow = spt.layers.planar_normalizing_flows(
                        config.nf_layers, name='posterior_flow')
                else:
                    self._posterior_flow = None

                # 2. بناء الـ VAE (القلب النابض للموديل)
                self._vae = VAE(
                    # P(z): الـ Prior - ربط النقاط زمنياً باستخدام LGSSM
                    p_z=TfpDistribution(
                        LinearGaussianStateSpaceModel(
                            num_timesteps=config.window_length,
                            transition_matrix=LinearOperatorIdentity(config.z_dim),
                            transition_noise=MultivariateNormalDiag(
                                scale_diag=tf.ones([config.z_dim])),
                            observation_matrix=LinearOperatorIdentity(config.z_dim),
                            observation_noise=MultivariateNormalDiag(
                                scale_diag=tf.ones([config.z_dim])),
                            initial_state_prior=MultivariateNormalDiag(
                                scale_diag=tf.ones([config.z_dim]))
                        )
                    ) if config.use_connected_z_p else Normal(mean=tf.zeros([config.z_dim]), 
                                                            std=tf.ones([config.z_dim])),
                    
                    # P(x|z): الـ Decoder
                    p_x_given_z=Normal,
                    
                    # Q(z|x): الـ Encoder (استخدام الـ RecurrentDistribution لربط الـ RNN بالـ VAE)
                    q_z_given_x=partial(
                        RecurrentDistribution,
                        mean_q_mlp=lambda x: tf.layers.dense(x, config.z_dim, 
                                        activation=None, name='z_mean'),
                        std_q_mlp=partial(softplus_std, units=config.z_dim, 
                                        epsilon=config.std_epsilon, name='z_std'),
                        z_dim=config.z_dim, 
                        window_length=config.window_length
                    ) if config.use_connected_z_q else Normal,
                    
                    # h_for_p_x: تغليف الـ Decoder بالـ RNN لفهم تسلسل حركة الروبوت
                    h_for_p_x=Lambda(
                        partial(
                            wrap_params_net,
                            h_for_dist=lambda x: rnn(x=x,
                                                    window_length=config.window_length,
                                                    rnn_num_hidden=config.rnn_num_hidden,
                                                    hidden_dense=2,
                                                    dense_dim=config.dense_dim,
                                                    name='rnn_p_x'),
                            mean_layer=lambda x: tf.layers.dense(x, config.x_dim, 
                                            name='x_mean'),
                            std_layer=partial(softplus_std, units=config.x_dim, 
                                            epsilon=config.std_epsilon, name='x_std')
                        ),
                        name='p_x_given_z'
                    ),
                    
                    # ─────────────────────────────────────────────────────────
                    # h_for_q_z: Encoder hidden network
                    #
                    # [CGAD] The raw input x is first routed through
                    # self._causal_process_x() which applies the
                    # CausalGraphModule (2-layer GCN on the 15-sensor causal
                    # graph) BEFORE the GRU encoder sees it.
                    # ─────────────────────────────────────────────────────────
                    # ─────────────────────────────────────────────────────────
                    # h_for_q_z: Hybrid Encoder (Y-Split)
                    #
                    # Branch A: GRU (Temporal Sequence)
                    # Branch B: AD Layer (Association Discrepancy)
                    # Both branches process the Entropy GCN output (x_causal).
                    # ─────────────────────────────────────────────────────────
                    h_for_q_z=Lambda(
                        lambda x: {
                            'input_q': self._hybrid_encoder(x)
                        },
                        name='q_z_given_x'
                    )
                )

    # =========================================================================
    # [CGAD] Causal graph processing for raw sensor input
    # =========================================================================
    def _causal_process_x(self, x):
        """
        Route the raw sensor input through the CausalGraphModule before
        handing it to the GRU encoder.

        The CausalGraphModule expects input of shape:
            [batch_size, num_sensors (15), features]
        but the OmniAnomaly pipeline provides x as:
            [batch_size, window_length, x_dim (15)]

        This method handles the reshape → GCN → reshape transformation.

        Shape walkthrough (with explicit comments):
        ──────────────────────────────────────────────
        1. x_input           : [batch_size, window_length, 15]
        2. batch_size         = tf.shape(x)[0]          (dynamic)
           window_length      = tf.shape(x)[1]          (dynamic)
           num_sensors        = 15                       (static / from config)
        3. Reshape to combine batch and window dims so
           the GCN sees each time-step independently:
              x_reshaped      : [batch_size * window_length, 15, 1]
        4. Pass through CausalGraphModule (2-layer GCN):
              x_causal        : [batch_size * window_length, 15, 1]
        5. Squeeze the trailing feature dim:
              x_squeezed      : [batch_size * window_length, 15]
        6. Reshape back to the original 3-D layout:
              x_out           : [batch_size, window_length, 15]
        """
        with tf.name_scope('causal_process_x'):
            # ── Step 1: Capture dynamic dimensions ───────────────────────────
            # x shape: [batch_size, window_length, 15]
            x_shape = tf.shape(x)
            batch_size = x_shape[0]
            window_length = x_shape[1]
            num_sensors = self._x_dims          # 15

            # ── Step 2: Reshape to [batch*window, 15, 1] ─────────────────────
            # Flatten batch and time so the GCN processes each timestep's
            # 15 sensor readings as an independent graph.
            x_reshaped = tf.reshape(x, [batch_size * window_length,
                                        num_sensors, 1])
            # x_reshaped shape: [batch_size * window_length, 15, 1]

            # ── Step 3: Apply 2-layer GCN (CausalGraphModule) ────────────────
            # The GCN multiplies by the normalized 15×15 adjacency matrix,
            # propagating causal information between sensors.
            x_causal = self.causal_input(x_reshaped)
            # x_causal shape: [batch_size * window_length, 15, 1]

            # ── Step 4: Squeeze the trailing feature dim ─────────────────────
            x_squeezed = tf.squeeze(x_causal, axis=-1)
            # x_squeezed shape: [batch_size * window_length, 15]

            # ── Step 5: Reshape back to [batch, window, 15] ──────────────────
            x_out = tf.reshape(x_squeezed, [batch_size, window_length,
                                            num_sensors])
            # x_out shape: [batch_size, window_length, 15]
            return x_out

    # =========================================================================
    # [HYBRID] Y-Split Architecture: RNN Branch + AD Branch
    # =========================================================================
    def _hybrid_encoder(self, x):
        """
        Implements the novel Y-split hybrid routing:
        1. Entropy Graph: self._causal_process_x(x)
        2. Branch A (GRU): rnn(x_causal)
        3. Branch B (AD): self._association_process_x(x_causal)
        4. Concatenation: tf.concat([h_gru, h_ad], axis=-1)
        """
        with tf.variable_scope('hybrid_encoder', reuse=tf.AUTO_REUSE):
            # Step 1: Spatial Context (Entropy GCN)
            x_causal = self._causal_process_x(x)
            
            # Step 2: Branch A - Temporal Sequence (GRU)
            # Returns: [batch, window, rnn_num_hidden]
            h_gru = rnn(
                x=x_causal,
                window_length=self.config.window_length,
                rnn_num_hidden=self.config.rnn_num_hidden,
                hidden_dense=2,
                dense_dim=self.config.dense_dim,
                name="rnn_q_z"
            )
            
            # Step 3: Branch B - Association Attention
            # Returns: [batch, window, dense_dim]
            h_ad = self._association_process_x(x_causal)
            
            # Step 4: Concatenation [batch, window, combined_dim]
            # combined_dim = rnn_num_hidden + dense_dim
            h_hybrid = tf.concat([h_gru, h_ad], axis=-1)
            
            # Print for dry-run verification (visible in console)
            print(f"[HYBRID] Branch A (GRU): {h_gru.shape}")
            print(f"[HYBRID] Branch B (AD):  {h_ad.shape}")
            print(f"[HYBRID] Final Concatenated: {h_hybrid.shape}")
            
            return h_hybrid

    # =========================================================================
    # [AD-VAE] Association Discrepancy processing for temporal modeling
    # =========================================================================
    def _association_process_x(self, x, reuse=tf.AUTO_REUSE):
        """
        Route features through Association Discrepancy layers (Wang & Zhang 2025).
        Shape walkthrough:
        1. x_in: [batch, window, features (15 or more)]
        2. Embedding + Positional Encoding -> [batch, window, dense_dim]
        3. AD-Layers (Prior P vs Sequence S) -> [batch, window, dense_dim]
        4. Capture symmetric KL discrepancy in self._discrepancy_list
        """
        with tf.variable_scope('association_process', reuse=tf.AUTO_REUSE):
            window_len = self._window_length
            dense_dim = self.config.dense_dim
            
            # Step 1: Base Embedding
            emb = tf.layers.dense(x, dense_dim, name="input_emb")
            
            # Step 2: Positional Encoding (Temporal context)
            pos = np.arange(window_len)[:, np.newaxis]
            div = np.exp(np.arange(0, dense_dim, 2) * -(np.log(10000.0) / dense_dim))
            pe_val = np.zeros((window_len, dense_dim), dtype=np.float32)
            pe_val[:, 0::2] = np.sin(pos * div)
            pe_val[:, 1::2] = np.cos(pos * div)
            pe = tf.constant(pe_val)
            
            h = tf.contrib.layers.layer_norm(emb + tf.expand_dims(pe, 0))
            
            # Step 3: AD Layers
            self._discrepancy_min_list = []
            self._discrepancy_max_list = []
            for i in range(self.config.adm_layers):
                h, d_min, d_max, S, P = association_layer(
                    x=h, 
                    heads=self.config.n_heads, 
                    d_model=dense_dim, 
                    name=f"AD_layer_{i}",
                    window_len=window_len
                )
                self._discrepancy_min_list.append(d_min) # Prior minimizes
                self._discrepancy_max_list.append(d_max) # Sequence maximizes
            
            return h


    @property
    def x_dims(self):
        """Get the number of `x` dimensions."""
        return self._x_dims

    @property
    def z_dims(self):
        """Get the number of `z` dimensions."""
        return self._z_dims


    @property
    def vae(self):
        """
        Get the VAE object of this :class:`OmniAnomaly` model.

        Returns:
            VAE: The VAE object of this model.
        """
        return self._vae


    @property
    def window_length(self):
        return self._window_length

    

    def get_training_loss(self, x, n_z=None):
        with tf.name_scope('training_loss'):
            chain = self.vae.chain(x, n_z=n_z, posterior_flow=self._posterior_flow)

            # ── z-samples from variational posterior ─────────────────────────
            # Shape: [n_z, batch, window, z_dim]  (or [batch, window, z_dim])
            z_samples = chain.variational['z'].tensor

            # Helper: normalize log-prob tensors to shape [n_z, batch]
            # so that all ELBO terms are broadcast-compatible.
            def _to_nz_batch(t):
                """Convert various possible shapes to [n_z, batch]."""
                has_samples = (len(z_samples.shape) == 4)
                if not has_samples:
                    # No explicit sampling dimension: just ensure leading sample axis.
                    if len(t.shape) == 1:
                        return tf.expand_dims(t, 0)  # [1, batch]
                    if len(t.shape) == 2:
                        return tf.expand_dims(tf.reduce_sum(t, axis=-1), 0)  # [1, batch]
                    if len(t.shape) >= 3:
                        t = tf.reduce_sum(t, axis=list(range(len(t.shape) - 1, 0, -1)))
                        return tf.expand_dims(t, 0)  # [1, batch]
                    return t

                # With samples: z_samples shape [n_z, batch, window, z_dim]
                nz = tf.shape(z_samples)[0]
                batch = tf.shape(z_samples)[1]
                window = tf.shape(z_samples)[2]

                t_rank = len(t.shape)

                # Case 1: rank-3, e.g. [n_z, batch, window]
                if t_rank == 3:
                    return tf.reduce_sum(t, axis=-1)  # [n_z, batch]

                # Case 2: rank-2: either [n_z, batch] already, or [batch, window]
                if t_rank == 2:
                    dyn_shape = tf.shape(t)

                    def from_batch_window():
                        t_bw = tf.reshape(t, [1, batch, window])      # [1, batch, window]
                        t_bw = tf.tile(t_bw, [nz, 1, 1])              # [n_z, batch, window]
                        return tf.reduce_sum(t_bw, axis=-1)           # [n_z, batch]

                    def as_is():
                        return t

                    is_batch_window = tf.logical_and(
                        tf.equal(dyn_shape[0], batch),
                        tf.equal(dyn_shape[1], window)
                    )
                    return tf.cond(is_batch_window, from_batch_window, as_is)

                # Case 3: rank-1: [batch]
                if t_rank == 1:
                    t = tf.reshape(t, [1, -1])              # [1, batch]
                    t = tf.tile(t, [nz, 1])                 # [n_z, batch]
                    return t

                # Fallback: reduce everything except (potential) [n_z, batch]
                while len(t.shape) > 2:
                    t = tf.reduce_sum(t, axis=-1)
                if len(t.shape) == 1:
                    t = tf.tile(tf.expand_dims(t, 0), [nz, 1])
                return t

            # ── log p(x|z): reconstruction log-likelihood ────────────────────
            log_px_z_raw = chain.model['x'].log_prob(group_ndims=1)
            log_px_z = _to_nz_batch(log_px_z_raw)

            # ── log q(z|x): variational posterior log-prob ───────────────────
            log_qz_x_raw = chain.variational['z'].log_prob(group_ndims=1)
            log_qz_x = _to_nz_batch(log_qz_x_raw)

            # ── log p(z): LGSSM (or Normal) prior from the VAE chain ─────────
            log_pz_raw = chain.model['z'].log_prob(group_ndims=1)
            log_pz = _to_nz_batch(log_pz_raw)

            # ── ELBO = E_q[log p(x|z)] - β * KL(q||p)  ─────────────────────
            kl_div   = log_qz_x - log_pz            # [n_z, batch]
            elbo     = log_px_z - self.beta * kl_div # [n_z, batch]
            total_loss = -tf.reduce_mean(elbo)       # scalar (minimise -ELBO)

            # ── L2 regularisation ─────────────────────────────────────────
            trainable_vars = tf.trainable_variables()
            l2_loss = (tf.add_n([tf.nn.l2_loss(v)
                                 for v in trainable_vars
                                 if 'bias' not in v.name])
                       if trainable_vars else tf.constant(0.0))
            total_loss += self.config.l2_reg * l2_loss

            # ── [AD-VAE] Association Discrepancy Loss ─────────────────────
            with tf.name_scope('reconstruction_discrepancy'):
                x_rec      = chain.model['x'].tensor
                window_len = self._window_length
                dim        = self.config.x_dim
                x_rec_flat = tf.reshape(x_rec, [-1, window_len, dim])
                _ = self._association_process_x(x_rec_flat, reuse=True)

            if self._discrepancy_min_list and self._discrepancy_max_list:
                d_min_sum  = tf.add_n(self._discrepancy_min_list) / len(self._discrepancy_min_list)
                d_max_sum  = tf.add_n(self._discrepancy_max_list) / len(self._discrepancy_max_list)
                total_loss += self.config.k_weight * tf.reduce_mean(d_min_sum - d_max_sum)

            return total_loss



    def get_score(self, x, n_z=None, last_point_only=True):
            with tf.name_scope('get_score'):
                # 1. طباعة توضيحية (لطيفة للمتابعة وقت الـ Testing)
                print('-' * 20, ' Testing with Root Cause Analysis ', '-' * 20)
                
                # 2. المرور عبر الـ Encoder (Variational) والـ Decoder (Model)
                # بنستخدم الـ posterior_flow عشان الـ Sampling بتاع Z يكون أدق ما يمكن
                q_net = self.vae.variational(x=x, n_z=n_z, posterior_flow=self._posterior_flow)
                p_net = self.vae.model(z=q_net['z'], x=x, n_z=n_z)
                
                # 3. استخراج وتحليل الـ Latent Variable (z)
                # z هي "البصمة" المختصرة لحالة الروبوت
                z_samples = q_net['z'].tensor
                
                if n_z is not None and n_z > 1:
                    # لو فيه Sampling، بنحسب المتوسط والانحراف المعياري عشان نفهم الـ Uncertainty
                    z_mean = tf.reduce_mean(z_samples, axis=0)
                    z_std = tf.math.reduce_std(z_samples, axis=0)
                else:
                    z_mean = z_samples
                    z_std = tf.zeros_like(z_mean)
                
                # دمج المتوسط والـ Std عشان يعبروا عن حالة الـ Latent space كاملة
                z_info = tf.concat((z_mean, z_std), axis=-1)

                # 4. حساب الـ Reconstruction Log Probability (الـ Score الحقيقي)
                # group_ndims=0: هي السر! بتخلينا نحسب الـ probability لكل سنسور لوحده
                # r_prob shape: (Batch_size, Window_length, X_dims)
                r_prob = p_net['x'].log_prob(group_ndims=0) 

                # 5. التركيز على آخر لحظة زمنية (Real-time detection)
                if last_point_only:
                    # بناخد آخر نقطة في الـ Window (اللي هي اللحظة الحالية)
                    # وبنحتفظ بكل الـ X_dims (السنسورات) عشان الـ RCA
                    r_prob = r_prob[:, -1, :] 
                    
                # [AD-VAE] Joint Anomaly Score (Wang & Zhang Eq. 16)
                # Softmax of negated discrepancy weights the raw reconstruction prob.
                if hasattr(self, '_discrepancy_min_list') and self._discrepancy_min_list:
                    # We use the mean discrepancy for scoring (from Prior branch)
                    dis_sum = tf.add_n(self._discrepancy_min_list) / len(self._discrepancy_min_list)
                    dis_last = dis_sum[:, -1] if last_point_only else dis_sum
                    dis_weight = tf.nn.softmax(-dis_last, axis=-1)
                    
                    # Scale r_prob (log-likelihood) by the temporal focus weight
                    r_prob = r_prob * tf.expand_dims(dis_weight, -1)
                    
                return r_prob, z_info
