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
        with tf.compat.v1.variable_scope(self._name):
            # ── Constant adjacency (created once, reused) ────────────────────
            # norm_A shape: [num_sensors, num_sensors]  (e.g. [15, 15])
            norm_A = tf.constant(self._norm_A, dtype=tf.float32, name='norm_A')

            # ── Layer 0: Linear projection  →  ReLU ─────────────────────────
            # x  shape: [batch, 15, in_channels]
            # W0 shape: [in_channels, 32]  (via tf.keras.layers.Dense)
            out = tf.keras.layers.Dense(32, activation=tf.nn.relu, name='W0')(x)
            # out shape: [batch, 15, 32]

            # ── Graph convolution: multiply by normalized adjacency ──────────
            # norm_A: [15, 15]   out: [batch, 15, 32]
            # Result: [batch, 15, 32]  –  each sensor aggregates its neighbours
            out = tf.einsum('ij,bjk->bik', norm_A, out)

            # ── Layer 1: Linear projection (no activation before residual) ───
            # W1 shape: [32, out_channels]
            out = tf.keras.layers.Dense(self._out_channels, name='W1')(out)
            # out shape: [batch, 15, out_channels]

            # ── Residual connection ──────────────────────────────────────────
            # out + x (requires in_channels == out_channels)
            return out + x


class OmniAnomaly(VarScopeObject):
    def __init__(self, config, name=None, scope=None):
            self._window_length = config.window_length
            self._x_dims = config.x_dim
            self.beta = getattr(config, 'beta', 1.0)
            self.config = config
            super(OmniAnomaly, self).__init__(name=name, scope=scope)
            
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
                        mean_q_mlp=lambda x: tf.keras.layers.Dense(units=config.z_dim, 
                                        activation=None, name='z_mean')(x),
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
                            mean_layer=lambda x: tf.keras.layers.Dense(units=config.x_dim, 
                                            name='x_mean')(x),
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
                    h_for_q_z=Lambda(
                        lambda x: {
                            'input_q': rnn(
                                x=self._causal_process_x(x),
                                window_length=config.window_length,
                                rnn_num_hidden=config.rnn_num_hidden,
                                hidden_dense=2,
                                dense_dim=config.dense_dim,
                                name="rnn_q_z")
                        },
                        name='q_z_given_x'
                    ) if config.use_connected_z_q else Lambda(
                        partial(
                            wrap_params_net,
                            h_for_dist=lambda x: rnn(
                                x=self._causal_process_x(x),
                                window_length=config.window_length,
                                rnn_num_hidden=config.rnn_num_hidden,
                                hidden_dense=2,
                                dense_dim=config.dense_dim,
                                name="rnn_q_z"),
                            mean_layer=lambda x: tf.keras.layers.Dense(units=config.z_dim, 
                                            name='z_mean')(x),
                            std_layer=partial(softplus_std, units=config.z_dim, 
                                            epsilon=config.std_epsilon, name='z_std')
                        ),
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
            
            # ── [BETA-VAE] Weighting the KL Divergence ───────────────────────────
            # We manually decompose ELBO to apply the beta weight to the KL term.
            # ELBO = E_q[log p(x|z)] - beta * KL(q(z|x) || p(z))
            log_px_z = chain.model['x'].log_prob()
            log_pz   = chain.model['z'].log_prob()
            log_qz_x = chain.variational['z'].log_prob()
            
            # negative_elbo = - (reconstruction_log_prob + log_prior - log_posterior)
            # We multiply (log_posterior - log_prior) by beta
            kl_div = log_qz_x - log_pz
            sgvb_loss = - (log_px_z - self.beta * kl_div)
            
            trainable_vars = tf.trainable_variables()
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in trainable_vars if 'bias' not in v.name])
            
            total_loss = tf.reduce_mean(sgvb_loss) + (self.config.l2_reg * l2_loss)
            
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
                    
                return r_prob, z_info
