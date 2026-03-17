# -*- coding: utf-8 -*-
import logging

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
from tfsnippet.distributions import Distribution


class TfpDistribution(Distribution):
    def __init__(self, distribution):
        if not isinstance(distribution, tfp.distributions.Distribution):
            raise TypeError('`distribution` is not an instance of `tfp.'
                            'distributions.Distribution`')
        self._distribution = distribution
        self._is_continuous = True
        self._is_reparameterized = self._distribution.reparameterization_type is tfp.distributions.FULLY_REPARAMETERIZED
        super(TfpDistribution, self).__init__(
            dtype=distribution.dtype,
            is_continuous=self._is_continuous,
            is_reparameterized=self._is_reparameterized,
            batch_shape=distribution.batch_shape,
            batch_static_shape=distribution.batch_shape,
            value_ndims=tf.size(distribution.event_shape) if distribution.event_shape else 0
        )

    @property
    def is_reparameterized(self):
        return self._is_reparameterized

    @property
    def is_continuous(self):
        return self._is_continuous

    @property
    def dtype(self):
        return self._distribution.dtype

    def sample(self, n_samples=None, is_reparameterized=None, group_ndims=0, compute_density=False, name=None):
        from tfsnippet.stochastic import StochasticTensor
        
        # تحسين 1: زيادة عدد الـ Samples يخلي لقط الـ Anomaly أدق
        if n_samples is None or n_samples < 5: 
            n_samples = 5 # رفعناها من 2 لـ 5 عشان الـ Importance Sampling
            
        with tf.name_scope(name=name, default_name='sample'):
            samples = self._distribution.sample(n_samples)
            
            # تحسين 2: بلاش reduce_mean فوراً! 
            # بنسيب الـ samples زي ما هي عشان الـ VAE ياخد "تنوع" في الآراء
            # المتوسط بيتحسب في الـ Loss function أحسن (IWAE logic)
            
            t = StochasticTensor(
                distribution=self,
                tensor=samples, # نبعت الـ Tensor كامل بالأبعاد بتاعته
                n_samples=n_samples,
                group_ndims=group_ndims,
                is_reparameterized=self.is_reparameterized
            )
            
            if compute_density:
                with tf.name_scope('compute_prob_and_log_prob'):
                    # تحسين 3: استخدام الـ Log_prob مباشرة لضمان الـ Stability
                    log_p = t.log_prob()
                    t._self_prob = tf.exp(log_p)
            return t

    def log_prob(self, given, group_ndims=0, name=None):
        with tf.name_scope(name=name, default_name='log_prob'):
            results = self._distribution.forward_filter(given)
            log_prob = results[0] if isinstance(results, (list, tuple)) else results

            # ── Shape normalisation ──────────────────────────────────────────
            # LGSSM forward_filter returns shapes like:
            #   [batch, window]       (no sampling)
            #   [n_z, batch, window]  (with n_z samples)
            # RecurrentDistribution.log_prob sums over z_dim AND time, returning:
            #   [n_z, batch]          (with n_z samples)
            # We align by summing over the trailing window dimension so both
            # have at most shape [n_z, batch].
            # Rank-3 → sum last axis  →  [n_z, batch]
            # Rank-2 → sum last axis  →  [batch]       (broadcasts fine against [n_z, batch])
            log_prob = tf.reduce_sum(log_prob, axis=-1)
            return log_prob



def softplus_std(inputs, units, epsilon, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # 1. طبقة الـ Dense مع Bias مدروس
        # استخدمنا kernel_initializer عشان نضمن إن البداية تكون مستقرة
        raw_std = tf.layers.dense(
            inputs,
            units, 
            name='dense',
            kernel_initializer=tf.glorot_uniform_initializer()
        )
        
        # 2. تطبيق الـ Softplus
        std = tf.nn.softplus(raw_std)
        
        # 3. [اللمسة العالمية] إضافة الـ Epsilon مع حماية إضافية
        # بنضمن إن الـ std ميكونش أصغر من الـ epsilon أبداً
        # وده بيمنع الـ Singular Matrices اللي بتوقف الموديل
        std = tf.maximum(std, epsilon) + 1e-8 
        
        return std

def rnn(x,
        window_length,
        rnn_num_hidden,
        rnn_cell='GRU',
        hidden_dense=2,
        dense_dim=200,
        time_axis=1,
        name='rnn'):
    from tensorflow.contrib import rnn as contrib_rnn
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if len(x.shape) == 4:
            x = tf.reduce_mean(x, axis=0)
        elif len(x.shape) != 3:
            logging.error("rnn input shape error")
        
        # 1. بناء الـ Stacked Cells
        rnn_cell_impl = tf.nn.rnn_cell
        def get_cell(num_units):
            if rnn_cell == 'LSTM':
                return rnn_cell_impl.BasicLSTMCell(num_units, forget_bias=1.0)
            elif rnn_cell == "GRU":
                return rnn_cell_impl.GRUCell(num_units)
            else:
                return rnn_cell_impl.BasicRNNCell(num_units)

        fw_cell = rnn_cell_impl.MultiRNNCell([get_cell(rnn_num_hidden) for _ in range(2)])
        bw_cell = rnn_cell_impl.MultiRNNCell([get_cell(rnn_num_hidden) for _ in range(2)])

        # 2. الـ Bidirectional Logic
        try:
            (outputs_fw, outputs_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                fw_cell, bw_cell, x, dtype=tf.float32)
            outputs = tf.concat([outputs_fw, outputs_bw], axis=-1)
        except Exception:
            x_unstacked = tf.unstack(x, window_length, time_axis)
            outputs, _, _ = contrib_rnn.static_bidirectional_rnn(
                fw_cell, bw_cell, x_unstacked, dtype=tf.float32)
            outputs = tf.stack(outputs, axis=time_axis)

        # 3. الـ Attention Mechanism (يتحط هنا قبل الـ Dense)
        # 
        attention_score = tf.layers.dense(outputs, 1, activation=tf.nn.tanh)
        attention_weights = tf.nn.softmax(attention_score, axis=1)
        outputs = outputs * attention_weights

        # 4. طبقات الـ Dense للتنعيم وضبط الأبعاد
        # أول طبقة بتقلل الحجم من (الضعف) لـ dense_dim عشان يرجع طبيعي
        for i in range(hidden_dense):
            outputs = tf.layers.dense(outputs, dense_dim, activation=tf.nn.relu)
            
        # سطر الأمان: بنخلي آخر مخرج بنفس حجم rnn_num_hidden عشان الـ VAE ميزعلش
        outputs = tf.layers.dense(outputs, rnn_num_hidden)
        
        return outputs



def wrap_params_net(inputs, h_for_dist, mean_layer, std_layer):
    with tf.variable_scope('hidden', reuse=tf.AUTO_REUSE):
        # 1. بنطلع الـ Features الأساسية من الـ RNN
        h = h_for_dist(inputs)
        
        # 2. تركة الـ Uncertainty (الـ Dropout الذكي)
        # بنخليه training=True عشان يفضل شغال في الـ Testing فيساعدنا نقيس "الشك"
        h = tf.nn.dropout(h, keep_prob=0.9) 

    # 3. حساب المتوسط والـ Std مع تفعيل الـ AUTO_REUSE للحفاظ على نفس الـ Scope (عشان الـ ValueError)
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        z_mean = mean_layer(h)
        
        # 4. حساب الـ Std باستخدام الـ "صمام الأمان" اللي لسه لاقيينه (softplus_std)
        # هنا بقى الذكاء.. بدل ما ننادي std_layer(h) مباشرة، 
        # إحنا هنمرر النتائج للـ softplus_std عشان نضمن إنها موجبة ومش بصفر
        
        # ملحوظة: units هنا هتكون هي الـ latent_dim (أبعاد الـ z)
        # و epsilon ده رقم صغير (زي 1e-4) عشان الأمان الرياضي
        z_std = softplus_std(h, units=z_mean.get_shape().as_list()[-1], epsilon=1e-4, name='z_std_secure')

    return {
        'mean': z_mean,
        'std': z_std,
    }

def wrap_params_net_srnn(inputs, h_for_dist):
    with tf.variable_scope('hidden', reuse=tf.AUTO_REUSE):
        # 1. استخراج الـ Features
        h = h_for_dist(inputs)
        
        # 2. [التحسين] إضافة Layer Norm (أحدث حاجة في 2025)
        # دي بتخلي الـ "input_q" قيمها مستقرة جداً مهما كانت السنسورات بتحدف أرقام بعيدة
        h = tf.contrib.layers.layer_norm(h)
        
        # 3. إضافة Dropout خفيف (لحماية الـ Stochasticity)
        h = tf.nn.dropout(h, keep_prob=0.95)

    return {
        'input_q': h
    }
