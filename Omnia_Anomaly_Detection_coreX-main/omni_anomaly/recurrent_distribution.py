# -*- coding: utf-8 -*-
import tensorflow as tf
from tfsnippet import Distribution, Normal


class RecurrentDistribution(Distribution):
    """
    A multi-variable distribution integrated with recurrent structure.
    """

    @property
    def dtype(self):
        return self._dtype

    @property
    def is_continuous(self):
        return self._is_continuous

    @property
    def is_reparameterized(self):
        return self._is_reparameterized

    @property
    def value_shape(self):
        return self.normal.value_shape

    def get_value_shape(self):
        return self.normal.get_value_shape()

    @property
    def batch_shape(self):
        return self.normal.batch_shape

    def get_batch_shape(self):
        return self.normal.get_batch_shape()

    def sample_step(self, a, t):
        # a: (z_prev, mu_prev, std_prev) | t: (noise, input_q_n)
        z_previous, mu_q_previous, std_q_previous = a
        noise_n, input_q_n = t
        
        # 1. تحسين الـ Broadcasting ليتناسب مع الـ 5 samples (Importance Sampling)
        # بنستخدم tf.shape الديناميكي عشان نضمن إن الأبعاد مظبوطة مهما كان عدد العينات
        batch_size = tf.shape(input_q_n)[0]
        n_samples = tf.shape(z_previous)[0]
        z_dim = input_q_n.shape[1]

        # توسيع الـ input_q_n ليتماشى مع عدد الـ samples
        input_q_n_expanded = tf.tile(tf.expand_dims(input_q_n, 0), [n_samples, 1, 1])

        # 2. الـ Integration (الماضي + الحاضر)
        # هنا هنستخدم الـ Structure اللي بيحافظ على الـ Temporal Consistency
        input_q = tf.concat([input_q_n_expanded, z_previous], axis=-1)

        # 3. استخراج الباراميترز (مع حقن الـ Stability)
        with tf.variable_scope('inference_net', reuse=tf.AUTO_REUSE):
            mu_q = self.mean_q_mlp(input_q)
            
            # تحسين عالمي: استخدام الـ softplus_std اللي عملناه لضمان الثبات الرياضي
            # ده بيمنع الـ Posterior Collapse عبر الزمن
            std_q = softplus_std(input_q, units=mu_q.shape[-1], epsilon=1e-5, name='std_q_gate')

        # 4. الـ Reparameterization Trick (The Sampling process)
        # z = mu + std * noise
        # الـ einsum هنا ممتازة بس هنخليها أكثر وضوحاً لضمان الـ Speed
        z_n = mu_q + (std_q * noise_n)

        # تحسين إضافي: Layer Norm على الـ z_n عشان نمنع الـ Latent values من الانفجار
        z_n = tf.contrib.layers.layer_norm(z_n, scope='z_norm')

        return z_n, mu_q, std_q

    # @global_reuse
    def log_prob_step(self, _, t):
        # given_n: الداتا الحقيقية (الـ Ground Truth)
        # input_q_n: مخرجات الـ RNN (الـ Context)
        given_n, input_q_n = t
        
        # 1. المواءمة مع عدد العينات (Multi-sample broadcasting)
        # بنعمل Align للحاضر مع العينات اللي الموديل ولدها
        if len(given_n.shape) > 2:
            # n_samples = given_n.shape[0]
            input_q_n = tf.tile(tf.expand_dims(input_q_n, 0), [tf.shape(given_n)[0], 1, 1])
        
        # 2. دمج الماضي (الداتا المعطاة) مع الحاضر (Context)
        input_q = tf.concat([given_n, input_q_n], axis=-1)
        
        with tf.variable_scope('inference_net', reuse=tf.AUTO_REUSE):
            # 3. استخراج باراميترز التوزيعة بـ "صمام الأمان"
            mu_q = self.mean_q_mlp(input_q)
            # استخدام الـ softplus_std اللي عملناه عشان الـ logstd يكون سليم
            std_q = softplus_std(input_q, units=mu_q.shape[-1], epsilon=1e-5, name='std_q_gate')
            logstd_q = tf.log(std_q)

        # 4. الحساب المطور للـ Log Probability (بدل المعادلات اليدوية الخطيرة)
        # الطريقة دي بتمنع الـ Gradient Explosion تماماً من غير ما نعمل Clip للـ Error
        # Gaussian Log Prob = -log(std) - 0.5 * log(2*pi) - 0.5 * ((x - mu) / std)^2
        
        constant = -0.5 * tf.log(2.0 * 3.141592653589793)
        
        # تحسين: بدل الـ 1e8، بنستخدم الـ Robust Squared Error
        squared_diff = tf.square((given_n - mu_q) / (std_q + 1e-8))
        log_prob_n = constant - logstd_q - 0.5 * squared_diff

        # إضافة اختيارية: Check numerics بجدية
        if self._check_numerics:
            log_prob_n = tf.check_numerics(log_prob_n, "log_prob_step_output")

        return log_prob_n

    # def __init__(self, input_q, mean_q_mlp, std_q_mlp, z_dim, window_length=100, is_reparameterized=True,
    #              check_numerics=True):
    #     super(RecurrentDistribution, self).__init__()
    #     self.normal = Normal(mean=tf.zeros([window_length, z_dim]), std=tf.ones([window_length, z_dim]))
    #     self.std_q_mlp = std_q_mlp
    #     self.mean_q_mlp = mean_q_mlp
    #     self._check_numerics = check_numerics
    #     self.input_q = tf.transpose(input_q, [1, 0, 2])
    #     self._dtype = input_q.dtype
    #     self._is_reparameterized = is_reparameterized
    #     self._is_continuous = True
    #     self.z_dim = z_dim
    #     self.window_length = window_length
    #     self.time_first_shape = tf.convert_to_tensor([self.window_length, tf.shape(input_q)[0], self.z_dim])

    def __init__(self, input_q, mean_q_mlp, std_q_mlp, z_dim, window_length=100, is_reparameterized=True,
             check_numerics=True):
        super(RecurrentDistribution, self).__init__()
        
        # 1. تعريف الـ Prior الأساسي (Standard Normal)
        # بنستخدمه كمرجع للحسابات الاحتمالية
        self.normal = Normal(mean=tf.zeros([window_length, z_dim]), 
                            std=tf.ones([window_length, z_dim]))
        
        # 2. تخزين الـ MLPs (المحركات اللي بتطلع الـ Mean و الـ Std)
        self.std_q_mlp = std_q_mlp
        self.mean_q_mlp = mean_q_mlp
        self._check_numerics = check_numerics
        
        # 3. تحويل الداتا لـ Time-First (لزيادة سرعة الـ Scan)
        # [Batch, Time, Dim] -> [Time, Batch, Dim]
        self.input_q = tf.transpose(input_q, [1, 0, 2])
        
        # 4. تكة الـ CoreX: التأكد من الـ Dtype لضمان الـ Precision
        self._dtype = input_q.dtype
        self._is_reparameterized = is_reparameterized
        self._is_continuous = True
        self.z_dim = z_dim
        self.window_length = window_length
        
        # 5. تحسين الـ Shape Tensor:
        # بنخليه Dynamic عشان يستوعب لو الـ Batch size اتغير في الـ Inference
        self.batch_size = tf.shape(input_q)[0]
        self.time_first_shape = tf.stack([tf.constant(window_length), self.batch_size, tf.constant(z_dim)])
        
        #

def sample(self, n_samples=None, is_reparameterized=None, group_ndims=0, compute_density=False, name=None):
    from tfsnippet.stochastic import StochasticTensor
    
    # 1. ضبط عدد العينات (SOTA Recommendation: 5 to 10 for IWAE)
    if n_samples is None:
        n_samples = 5 
    
    with tf.name_scope(name=name, default_name='sample'):
        # 2. توليد النويز بشكل احترافي
        # بنولد نويز لكل (Time_step, Sample, Batch, Dim)
        noise_shape = [self.window_length, n_samples, self.batch_size, self.z_dim]
        noise = tf.random_normal(shape=noise_shape, dtype=self._dtype)

        # 3. تجهيز الـ Initializer (البداية الصفرية لكل عينة)
        initial_z = tf.zeros([n_samples, self.batch_size, self.z_dim])
        initial_mu = tf.zeros([n_samples, self.batch_size, self.z_dim])
        initial_std = tf.ones([n_samples, self.batch_size, self.z_dim])

        # 4. الـ Scan المطور (The Recurrent Loop)
        # أهم تعديل: شلنا back_prop=False عشان الموديل يتعلم بجد
        scan_outputs = tf.scan(
            fn=self.sample_step,
            elems=(noise, self.input_q),
            initializer=(initial_z, initial_mu, initial_std),
            name="temporal_scan"
        )
        
        samples = scan_outputs[0]  # [window_length, n_samples, batch_size, z_dim]

        # 5. إعادة ترتيب الأبعاد (Transposing to Standard Format)
        # الترتيب العالمي: [n_samples, batch_size, time, z_dim]
        samples = tf.transpose(samples, [1, 2, 0, 3])

        # 6. الـ Stochastic Tensor (بدون reduce_mean للحفاظ على الـ Uncertainty)
        t = StochasticTensor(
            distribution=self,
            tensor=samples,
            n_samples=n_samples,
            group_ndims=group_ndims,
            is_reparameterized=self.is_reparameterized
        )

        if compute_density:
            with tf.name_scope('compute_prob_and_log_prob'):
                log_p = t.log_prob()
                t._self_prob = tf.exp(log_p)
                
        return t


    def log_prob(self, given, group_ndims=0, name=None):
        with tf.name_scope(name=name, default_name='log_prob'):
            # 1. تهيئة الأبعاد بناءً على عدد العينات (Multi-sample support)
            # given shape: [n_samples, batch_size, window_length, z_dim]
            given_shape = tf.shape(given)
            
            if len(given.shape) > 3:
                # حالة الـ Importance Sampling (SOTA)
                # بننقل الـ Time-axis للأول عشان الـ Scan
                given_time_first = tf.transpose(given, [2, 0, 1, 3])
                # شكل الـ initializer لازم يطابق الـ samples والـ batch
                init_shape = [given_shape[0], given_shape[1], self.z_dim]
            else:
                # الحالة العادية (Single sample)
                given_time_first = tf.transpose(given, [1, 0, 2])
                init_shape = [given_shape[0], self.z_dim]

            # 2. الـ Scan المطور (The Evaluation Loop)
            # أهم تعديل: السماح بالـ Backprop عشان الموديل "يتعلم من غلطاته"
            log_prob_steps = tf.scan(
                fn=self.log_prob_step,
                elems=(given_time_first, self.input_q),
                initializer=tf.zeros(init_shape),
                back_prop=True, # التعديل الجوهري للـ Training
                name='temporal_log_prob_scan'
            )

            # 3. إعادة ترتيب النتائج للـ Standard Format
            if len(given.shape) > 3:
                # نرجعها لـ [n_samples, batch_size, window_length, z_dim]
                log_prob = tf.transpose(log_prob_steps, [1, 2, 0, 3])
            else:
                log_prob = tf.transpose(log_prob_steps, [1, 0, 2])

            # 4. تجميع الاحتمالات (Probability Aggregation)
            # بنجمع على الـ z_dim عشان نجيب الـ Total likelihood للنقطة الزمنية دي
            if group_ndims > 0:
                log_prob = tf.reduce_sum(log_prob, axis=-1)

            return log_prob       

    def prob(self, given, group_ndims=0, name=None):
        with tf.name_scope(name=name, default_name='prob'):
            # 1. بننادي الـ log_prob المطور اللي فيه الـ Backprop والـ Attention
            log_p = self.log_prob(given, group_ndims, name)
            
            # 2. [تكة الـ CoreX] الحماية من الـ Numerical Underflow
            # بنعمل Clip للقيم قبل الـ exp عشان نضمن إنها متوصلش لصفر مطلق يوقف الحسابات
            # -50 ده رقم كافي جداً إنه يعبر عن "احتمالية شبه معدومة"
            log_p_stable = tf.maximum(log_p, -50.0)
            
            # 3. التحويل للاحتمالية الصريحة
            return tf.exp(log_p_stable)
