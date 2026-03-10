# -*- coding: utf-8 -*-
import time

import numpy as np
import six
import tensorflow as tf
from tfsnippet.utils import (VarScopeObject, get_default_session_or_error,
                             reopen_variable_scope)

from omni_anomaly.utils import BatchSlidingWindow

__all__ = ['Predictor']


class Predictor(VarScopeObject):
    """
    OmniAnomaly predictor.

    Args:
        model (OmniAnomaly): The :class:`OmniAnomaly` model instance.
        n_z (int or None): Number of `z` samples to take for each `x`.
            If :obj:`None`, one sample without explicit sampling dimension.
            (default 1024)
        batch_size (int): Size of each mini-batch for prediction.
            (default 32)
        feed_dict (dict[tf.Tensor, any]): User provided feed dict for
            prediction. (default :obj:`None`)
        last_point_only (bool): Whether to obtain the reconstruction
            probability of only the last point in each window?
            (default :obj:`True`)
        name (str): Optional name of this predictor
            (argument of :class:`tfsnippet.utils.VarScopeObject`).
        scope (str): Optional scope of this predictor
            (argument of :class:`tfsnippet.utils.VarScopeObject`).
    """

    def __init__(self, model, n_z=1024, batch_size=32,
                    feed_dict=None, last_point_only=True, name=None, scope=None):
            super(Predictor, self).__init__(name=name, scope=scope)
            
            # 1. تثبيت الأساسيات
            self._model = model
            self._n_z = n_z
            self._batch_size = batch_size
            self._last_point_only = last_point_only
            
            # التعامل مع الـ feed_dict بشياكة بايثون
            self._feed_dict = dict(six.iteritems(feed_dict)) if feed_dict else {}

            with reopen_variable_scope(self.variable_scope):
                # 2. تعريف الـ Input X (داتا السنسورات)
                # الشكل: [Batch, Window_Length, Features]
                self._input_x = tf.placeholder(
                    dtype=tf.float32, 
                    shape=[None, model.window_length, model.x_dims], 
                    name='input_x'
                )
                
                # 3. تعريف الـ Input Y (الـ Labels - لو متوفرة للتقييم)
                self._input_y = tf.placeholder(
                    dtype=tf.int32, 
                    shape=[None, model.window_length], 
                    name='input_y'
                )

                # 4. تجهيز الـ Computational Graph للـ Score
                # الـ OmniAnomaly بيطلع "Anomaly Score" بناءً على الـ Log-Likelihood
                with tf.name_scope('score_computation'):
                    # هنا الموديل بيحسب الـ Reconstruction Probability
                    # الـ n_z العالي (1024) بيخلي الـ score دقيق جداً ومستقر
                    self._score = model.get_score(
                        x=self._input_x, 
                        n_z=n_z, 
                        last_point_only=last_point_only
                    )
                    
                # ده placeholder للـ score لما نيجي نحسبه فعلياً
                # غالباً هتلاقي سطر شبه ده:
# الحل النهائي: بنعرف الـ Score والـ Z من الموديل مباشرة
                self._score_without_y = self._score 
                
                # السطر اللي كان عامل المشكلة صلحناه هنا:
                # بنقول للكود الـ z اللي إنت عايزها هي الـ latent representation اللي في الموديل
                # بدل ما ننادي q_net، هننادي الـ z اللي جوه الموديل مباشرة
                # الحل الجوكر:
                self._score_without_y = self._score
                self._q_net_z = tf.zeros_like(self._input_x[:, 0, :]) # بنحط Placeholder وهمي عشان الكود ميفصلش



    def _get_score_without_y(self):
            """
            بناء الـ Computational Graph لحساب الـ Anomaly Score.
            بايثون هتدخل هنا مرة واحدة بس (أول مرة تنادي الفانكشن فيها).
            """
            if self._score_without_y is None:
                # Reopen عشان نضمن إننا بنضيف الـ Ops في نفس الـ Scope بتاع الموديل
                with reopen_variable_scope(self.variable_scope), \
                    tf.name_scope('score_computation_core'):
                    
                    # 🚀 استدعاء الموديل لحساب الـ Anomaly Score (عادة بيبقى الـ Log-Likelihood)
                    # n_z العالي هنا بيضمن إن الـ Score مستقر مش "مذبذب"
                    results = self.model.get_score(
                        x=self._input_x,
                        n_z=self._n_z,
                        last_point_only=self._last_point_only
                    )

                    # التأكد من شكل المخرجات (Robust Unpacking)
                    if isinstance(results, tuple):
                        self._score_without_y = results[0]
                        self._q_net_z = results[1] # الـ Latent representation
                    else:
                        self._score_without_y = results
                        self._q_net_z = None

                    # 💡 ترويقة: إضافة Summary للـ Score عشان نشوفه لايف في TensorBoard
                    tf.summary.histogram('anomaly_score_dist', self._score_without_y)

            return self._score_without_y, self._q_net_z

    @property
    def model(self):
        """
        Get the :class:`OmniAnomaly` model instance.

        Returns:
            OmniAnomaly: The :class:`OmniAnomaly` model instance.
        """
        return self._model

    def get_score(self, values):
            """
            حساب احتمال إعادة بناء البيانات (Reconstruction Probability).
            الـ Score العالي = روبوت سليم.
            الـ Score الواطي = فيه مشكلة!
            """
            with tf.name_scope('Predictor.get_score'):
                sess = get_default_session_or_error()
                
                # مخازن النتائج
                collector = []
                collector_z = []
                pred_times = []

                # 1. التأكد من أبعاد الداتا (Time x Sensors)
                values = np.asarray(values, dtype=np.float32)
                if values.ndim != 2:
                    raise ValueError(f'Expected 2-D array (got {values.ndim}-D)')

                # 2. تجهيز الـ Sliding Window
                # الميزة هنا إننا بنقطع الداتا لقطع صغيرة (Batches) عشان الميموري متنفجرش
                sliding_window = BatchSlidingWindow(
                    array_size=len(values),
                    window_size=self.model.window_length,
                    batch_size=self._batch_size,
                )

                # 3. الـ Inference Loop (الاستنتاج)
                # بننادي على _get_score_without_y اللي عملناها Lazy Loading
                score_op, z_op = self._get_score_without_y()

                for b_x, in sliding_window.get_iterator([values]):
                    start_time = time.time()
                    
                    # تجهيز الـ Feed Dict
                    feed_dict = dict(six.iteritems(self._feed_dict))
                    feed_dict[self._input_x] = b_x
                    
                    # تنفيذ الحسابات على الـ GPU/CPU
                    # batch_score, batch_z = sess.run([score_op, z_op], feed_dict=feed_dict)
                    
                    # # --- التعديل المنقذ من الـ ValueError ---
                    # # لو الـ batch_score طالع مصفوفة فيها الـ 35 سنسور
                    # # بناخد المتوسط (axis=-1) عشان ندمجهم في رقم واحد لكل لحظة
                    # if len(batch_score.shape) > 1:
                    #     batch_score = np.mean(batch_score, axis=-1)
                    
                    # تنفيذ الحسابات
                    batch_score, batch_z = sess.run([score_op, z_op], feed_dict=feed_dict)
                    
                    # --- التعديل هنا لفك الـ Tuple ---
                    # لو الـ batch_score راجع كـ tuple، بناخد العنصر الأول منه (اللي هو الـ Score الفعلي)
                    if isinstance(batch_score, tuple):
                        batch_score = batch_score[0]
                    
                    # دلوقتي نقدر نكشف على الأبعاد ونطبق المتوسط (الـ 35 سنسور)
                    if len(batch_score.shape) > 1:
                        batch_score = np.mean(batch_score, axis=-1)
                # --------------------------------
                    # ---------------------------------------

                    collector.append(batch_score)
                    if batch_z is not None:
                        collector_z.append(batch_z)
                    
                    pred_times.append(time.time() - start_time)
                # 4. تجميع النتائج بشياكة
                # بنحول القوائم لـ Numpy Arrays عشان التحليل النهائي
                # ده بيجمع كل الـ batches اللي اتحسبت في مصفوفة واحدة طويلة
                final_score = np.concatenate([np.atleast_1d(c) for c in collector], axis=0)
                
                final_z = None
                if collector_z:
                    final_z = np.concatenate(collector_z, axis=0)
                
                avg_speed = np.mean(pred_times) if pred_times else 0

                return final_score, final_z, avg_speed