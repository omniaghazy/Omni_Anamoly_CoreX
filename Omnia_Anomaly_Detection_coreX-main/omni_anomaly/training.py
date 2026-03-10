# -*- coding: utf-8 -*-
import time

import numpy as np
import six
import tensorflow as tf
from tfsnippet.scaffold import TrainLoop
from tfsnippet.shortcuts import VarScopeObject
from tfsnippet.utils import (reopen_variable_scope,
                             get_default_session_or_error,
                             ensure_variables_initialized,
                             get_variables_as_dict)

from omni_anomaly.utils import BatchSlidingWindow

__all__ = ['Trainer']


class Trainer(VarScopeObject):
    """
    OmniAnomaly trainer.

    Args:
        model (OmniAnomaly): The :class:`OmniAnomaly` model instance.
        model_vs (str or tf.VariableScope): If specified, will collect
            trainable variables only from this scope.  If :obj:`None`,
            will collect all trainable variables within current graph.
            (default :obj:`None`)
        n_z (int or None): Number of `z` samples to take for each `x`.
            (default :obj:`None`, one sample without explicit sampling
            dimension)
        feed_dict (dict[tf.Tensor, any]): User provided feed dict for
            training. (default :obj:`None`, indicating no feeding)
        valid_feed_dict (dict[tf.Tensor, any]): User provided feed dict for
            validation.  If :obj:`None`, follow `feed_dict` of training.
            (default :obj:`None`)
        use_regularization_loss (bool): Whether or not to add regularization
            loss from `tf.GraphKeys.REGULARIZATION_LOSSES` to the training
            loss? (default :obj:`True`)
        max_epoch (int or None): Maximum epochs to run.  If :obj:`None`,
            will not stop at any particular epoch. (default 256)
        max_step (int or None): Maximum steps to run.  If :obj:`None`,
            will not stop at any particular step.  At least one of `max_epoch`
            and `max_step` should be specified. (default :obj:`None`)
        batch_size (int): Size of mini-batches for training. (default 256)
        valid_batch_size (int): Size of mini-batches for validation.
            (default 1024)
        valid_step_freq (int): Run validation after every `valid_step_freq`
            number of training steps. (default 100)
        initial_lr (float): Initial learning rate. (default 0.001)
        lr_anneal_epochs (int): Anneal the learning rate after every
            `lr_anneal_epochs` number of epochs. (default 10)
        lr_anneal_factor (float): Anneal the learning rate with this
            discount factor, i.e., ``learning_rate = learning_rate
            * lr_anneal_factor``. (default 0.75)
        optimizer (Type[tf.train.Optimizer]): The class of TensorFlow
            optimizer. (default :class:`tf.train.AdamOptimizer`)
        optimizer_params (dict[str, any] or None): The named arguments
            for constructing the optimizer. (default :obj:`None`)
        grad_clip_norm (float or None): Clip gradient by this norm.
            If :obj:`None`, disable gradient clip by norm. (default 10.0)
        check_numerics (bool): Whether or not to add TensorFlow assertions
            for numerical issues? (default :obj:`True`)
        name (str): Optional name of this trainer
            (argument of :class:`tfsnippet.utils.VarScopeObject`).
        scope (str): Optional scope of this trainer
            (argument of :class:`tfsnippet.utils.VarScopeObject`).
    """

    def __init__(self, model, model_vs=None, n_z=None,
                    feed_dict=None, valid_feed_dict=None,
                    use_regularization_loss=True,
                    max_epoch=256, max_step=None, batch_size=256,
                    valid_batch_size=1024, valid_step_freq=100,
                    initial_lr=0.001, lr_anneal_epochs=10, lr_anneal_factor=0.75,
                    optimizer=tf.train.AdamOptimizer, optimizer_params=None,
                    grad_clip_norm=10.0,  # 🔥 ترويقة 1: قللنا الـ clip لاستقرار الـ GRU
                    check_numerics=True,
                    name=None, scope=None):
            super(Trainer, self).__init__(name=name, scope=scope)

            # [تخزين الـ arguments - زي ما هي]
            self._model = model
            self._n_z = n_z
            self._feed_dict = dict(six.iteritems(feed_dict)) if feed_dict else {}
            self._valid_feed_dict = dict(six.iteritems(valid_feed_dict)) if valid_feed_dict else self._feed_dict
            
            self._max_epoch = max_epoch
            self._max_step = max_step
            self._batch_size = batch_size
            self._valid_batch_size = valid_batch_size
            self._valid_step_freq = valid_step_freq
            self._initial_lr = initial_lr
            self._lr_anneal_epochs = lr_anneal_epochs
            self._lr_anneal_factor = lr_anneal_factor

            with reopen_variable_scope(self.variable_scope):
                # الـ Global Step
                self._global_step = tf.get_variable(
                    dtype=tf.int64, name='global_step', trainable=False,
                    initializer=tf.constant(0, dtype=tf.int64)
                )

                # 🔥 ترويقة 2: الـ Input X لازم يطابق أبعاد الـ Robot Sensors
                self._input_x = tf.placeholder(
                    dtype=tf.float32, 
                    shape=[None, model.window_length, model.x_dims], 
                    name='input_x'
                )
                
                # الـ Learning Rate Placeholder
                self._learning_rate = tf.placeholder(
                    dtype=tf.float32, shape=(), name='learning_rate_placeholder'
                )

                # حساب الـ Loss
                with tf.name_scope('loss'):
                    # الموديل هنا بيحسب الـ ELBO (Evidence Lower Bound)
                    loss = model.get_training_loss(x=self._input_x, n_z=n_z)
                    if use_regularization_loss:
                        loss += tf.losses.get_regularization_loss()
                    self._loss = loss

                # تجميع المتغيرات اللي هتتمرن
                train_params = get_variables_as_dict(
                    scope=model_vs, collection=tf.GraphKeys.TRAINABLE_VARIABLES)
                self._train_params = train_params

                # تجهيز الـ Optimizer
                opt_params = dict(six.iteritems(optimizer_params)) if optimizer_params else {}
                opt_params['learning_rate'] = self._learning_rate
                self._optimizer = optimizer(**opt_params)

                # 🔥 ترويقة 3: الـ Gradient Clipping الاحترافي
                origin_grad_vars = self._optimizer.compute_gradients(
                    self._loss, list(six.itervalues(self._train_params))
                )
                grad_vars = []
                for grad, var in origin_grad_vars:
                    if grad is not None:
                        # Clipping by norm بيحافظ على اتجاه الـ gradient بس بيصغر طوله
                        if grad_clip_norm:
                            grad = tf.clip_by_norm(grad, grad_clip_norm)
                        if check_numerics:
                            grad = tf.check_numerics(grad, f'numeric_issue_{var.name}')
                        grad_vars.append((grad, var))

                # الـ Training Op مع الـ Control Dependencies عشان الـ Batch Norm لو موجود
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    self._train_op = self._optimizer.apply_gradients(
                        grad_vars, global_step=self._global_step)

                # 🔥 ترويقة 4: Summaries أذكى لمراقبة التدريب في TensorBoard
                with tf.name_scope('summary'):
                    summaries = [
                        tf.summary.scalar('total_loss', self._loss),
                        tf.summary.scalar('learning_rate', self._learning_rate)
                    ]
                    # إحصائيات للأوزان (عشان نتأكد إن مفيش Vanishing Gradients)
                    for v in six.itervalues(self._train_params):
                        summaries.append(tf.summary.histogram(v.name.replace(':', '_'), v))
                    self._summary_op = tf.summary.merge(summaries)

                # الـ Initializer
                self._trainer_initializer = tf.variables_initializer(
                    list(six.itervalues(get_variables_as_dict(scope=self.variable_scope, 
                                                            collection=tf.GraphKeys.GLOBAL_VARIABLES)))
                )


    @property
    def model(self):
        """
        Get the :class:`OmniAnomaly` model instance.

        Returns:
            OmniAnomaly: The :class:`OmniAnomaly` model instance.
        """
        return self._model

    def fit(self, values, valid_portion=0.3, summary_dir=None):
            sess = get_default_session_or_error()

            # تجهيز الداتا
            values = np.asarray(values, dtype=np.float32)
            if len(values.shape) != 2:
                raise ValueError('`values` must be a 2-D array (Time, Features)')

            # تقسيم الداتا (Train/Valid)
            n = int(len(values) * valid_portion)
            train_values, v_x = values[:-n], values[-n:]

            # بناء الشبابيك (Sliding Windows)
            train_sliding_window = BatchSlidingWindow(
                array_size=len(train_values),
                window_size=self.model.window_length,
                batch_size=self._batch_size,
                shuffle=True,
                ignore_incomplete_batch=True,
            )
            valid_sliding_window = BatchSlidingWindow(
                array_size=len(v_x),
                window_size=self.model.window_length,
                batch_size=self._valid_batch_size,
            )

            # تهيئة المتغيرات
            sess.run(self._trainer_initializer)
            ensure_variables_initialized(self._train_params)

            lr = self._initial_lr
            with TrainLoop(
                    param_vars=self._train_params,
                    early_stopping=True,
                    summary_dir=summary_dir,
                    max_epoch=self._max_epoch,
                    max_step=self._max_step) as loop:
                
                loop.print_training_summary()
                train_batch_time, valid_batch_time = [], []

                for epoch in loop.iter_epochs():
                    train_iterator = train_sliding_window.get_iterator([train_values])
                    # ✅ التعديل: عرفنا الـ epoch_start_time هنا عشان الحسابات تبقى دقيقة
                    epoch_start_time = time.time()
                    
                    for step, (batch_x,) in loop.iter_steps(train_iterator):
                        start_batch_time = time.time()
                        
                        feed_dict = dict(six.iteritems(self._feed_dict))
                        feed_dict[self._learning_rate] = lr
                        feed_dict[self._input_x] = batch_x
                        
                        loss, _ = sess.run([self._loss, self._train_op], feed_dict=feed_dict)
                        
                        loop.collect_metrics({'loss': loss})
                        train_batch_time.append(time.time() - start_batch_time)

                        if step % self._valid_step_freq == 0:
                            # حساب الوقت المستغرق في التدريب حتى الآن
                            train_duration = time.time() - epoch_start_time
                            loop.collect_metrics({'train_time': train_duration})

                            if summary_dir:
                                loop.add_summary(sess.run(self._summary_op, feed_dict=feed_dict))

                            with loop.timeit('valid_time'), loop.metric_collector('valid_loss') as mc:
                                v_it = valid_sliding_window.get_iterator([v_x])
                                for (b_v_x,) in v_it:
                                    v_start = time.time()
                                    v_feed = dict(six.iteritems(self._valid_feed_dict))
                                    v_feed[self._input_x] = b_v_x
                                    v_loss = sess.run(self._loss, feed_dict=v_feed)
                                    valid_batch_time.append(time.time() - v_start)
                                    mc.collect(v_loss, weight=len(b_v_x))

                            loop.print_logs()
                            # إعادة تصفير وقت البداية بعد الـ validation عشان الـ log الجاي
                            epoch_start_time = time.time()

                    # الـ Learning Rate Decay
                    if self._lr_anneal_epochs and epoch % self._lr_anneal_epochs == 0:
                        lr *= self._lr_anneal_factor
                        loop.println(f'🚀 [Update] Epoch {epoch}: LR reduced to {lr:.6f}')

                return {
                    'best_valid_loss': float(loop.best_valid_metric),
                    'train_step_avg_time': np.mean(train_batch_time),
                    'valid_step_avg_time': np.mean(valid_batch_time),
                }