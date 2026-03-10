# -*- coding: utf-8 -*-
from functools import partial

import tensorflow as tf
import tfsnippet as spt
from tensorflow.python.ops.linalg.linear_operator_identity import LinearOperatorIdentity
from tensorflow_probability.python.distributions import LinearGaussianStateSpaceModel, MultivariateNormalDiag
from tfsnippet.distributions import Normal
from tfsnippet.utils import VarScopeObject, reopen_variable_scope
from tfsnippet.variational import VariationalInference

from omni_anomaly.recurrent_distribution import RecurrentDistribution
from omni_anomaly.vae import Lambda, VAE
from omni_anomaly.wrapper import TfpDistribution, softplus_std, rnn, wrap_params_net

class OmniAnomaly(VarScopeObject):
    def __init__(self, config, name=None, scope=None):
            self._window_length = config.window_length
            self._x_dims = config.x_dim
            self.beta = getattr(config, 'beta', 1.0)
            self.config = config
            super(OmniAnomaly, self).__init__(name=name, scope=scope)
            
            with reopen_variable_scope(self.variable_scope):
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
                        mean_q_mlp=partial(tf.layers.dense, units=config.z_dim, 
                                        activation=None, name='z_mean', reuse=tf.AUTO_REUSE),
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
                            mean_layer=partial(tf.layers.dense, units=config.x_dim, 
                                            name='x_mean', reuse=tf.AUTO_REUSE),
                            std_layer=partial(softplus_std, units=config.x_dim, 
                                            epsilon=config.std_epsilon, name='x_std')
                        ),
                        name='p_x_given_z'
                    ),
                    
                    # h_for_q_z: تغليف الـ Encoder بالـ RNN
                    h_for_q_z=Lambda(
                        lambda x: {
                            'input_q': rnn(x=x,
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
                            h_for_dist=lambda x: rnn(x=x,
                                                    window_length=config.window_length,
                                                    rnn_num_hidden=config.rnn_num_hidden,
                                                    hidden_dense=2,
                                                    dense_dim=config.dense_dim,
                                                    name="rnn_q_z"),
                            mean_layer=partial(tf.layers.dense, units=config.z_dim, 
                                            name='z_mean', reuse=tf.AUTO_REUSE),
                            std_layer=partial(softplus_std, units=config.z_dim, 
                                            epsilon=config.std_epsilon, name='z_std')
                        ),
                        name='q_z_given_x'
                    )
                )


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