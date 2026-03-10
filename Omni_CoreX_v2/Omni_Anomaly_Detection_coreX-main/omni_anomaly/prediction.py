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
    OmniAnomaly predictor. Runs batched inference to produce per-timestep
    anomaly scores (sum of per-sensor log-probabilities).

    Args:
        model (OmniAnomaly): The model instance.
        n_z (int): Number of z samples per window (higher = more stable score).
        batch_size (int): Inference batch size.
        feed_dict (dict): Extra TF feed entries.
        last_point_only (bool): Score only the last point in each window.
    """

    def __init__(self, model, n_z=1024, batch_size=32,
                 feed_dict=None, last_point_only=True, name=None, scope=None):
        super(Predictor, self).__init__(name=name, scope=scope)

        self._model = model
        self._n_z = n_z
        self._batch_size = batch_size
        self._last_point_only = last_point_only
        self._feed_dict = dict(six.iteritems(feed_dict)) if feed_dict else {}

        # Lazy-initialised TF tensors (built on first call to get_score)
        self._input_x = None
        self._score_op = None
        self._z_op = None

    def _build_graph(self):
        """Build the TF inference graph (called once, lazily)."""
        with reopen_variable_scope(self.variable_scope):
            # Input placeholder: [Batch, Window, Sensors]
            self._input_x = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self._model.window_length, self._model.x_dims],
                name='input_x'
            )

            with tf.name_scope('score_computation'):
                results = self._model.get_score(
                    x=self._input_x,
                    n_z=self._n_z,
                    last_point_only=self._last_point_only
                )

            # get_score returns (r_prob, z_info)
            if isinstance(results, (tuple, list)):
                self._score_op = results[0]   # shape: (Batch, X_dims)
                self._z_op = results[1]        # shape: (Batch, 2*z_dim)
            else:
                self._score_op = results
                self._z_op = None

    @property
    def model(self):
        return self._model

    def get_score(self, values):
        """
        Compute reconstruction log-probability (anomaly score) for `values`.

        Score semantics:
          - High (less negative) = normal
          - Low (very negative)  = anomaly

        Returns:
            final_score : np.ndarray, shape (N_windows,)
                Sum of per-sensor log-probs for each timestep.
            final_z     : np.ndarray or None
                Latent encodings (for RCA / visualisation).
            avg_speed   : float
                Average seconds per batch.
        """
        # Build graph on first call
        if self._input_x is None:
            self._build_graph()

        sess = get_default_session_or_error()

        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f'Expected 2-D array (Time x Sensors), got {values.ndim}-D')

        sliding_window = BatchSlidingWindow(
            array_size=len(values),
            window_size=self._model.window_length,
            batch_size=self._batch_size,
        )

        collector_score = []
        collector_z = []
        pred_times = []

        for (b_x,) in sliding_window.get_iterator([values]):
            t0 = time.time()

            feed_dict = dict(six.iteritems(self._feed_dict))
            feed_dict[self._input_x] = b_x

            # Run inference
            if self._z_op is not None:
                batch_score, batch_z = sess.run(
                    [self._score_op, self._z_op], feed_dict=feed_dict)
            else:
                batch_score = sess.run(self._score_op, feed_dict=feed_dict)
                batch_z = None

            # ── [FIX] Collapse per-sensor log-probs to one score per timestep ──
            # r_prob shape from model: (Batch, X_dims)  [last_point_only=True]
            # We SUM log-probs (joint log-likelihood) rather than mean,
            # because joint prob = product of independent probs → sum of logs.
            if isinstance(batch_score, (tuple, list)):
                batch_score = batch_score[0]
            if batch_score.ndim > 1:
                batch_score = np.sum(batch_score, axis=-1)   # (Batch,)

            collector_score.append(np.atleast_1d(batch_score))
            if batch_z is not None:
                collector_z.append(batch_z)

            pred_times.append(time.time() - t0)

        # Concatenate batches
        final_score = np.concatenate(collector_score, axis=0)   # (N_windows,)
        final_z = np.concatenate(collector_z, axis=0) if collector_z else None
        avg_speed = float(np.mean(pred_times)) if pred_times else 0.0

        return final_score, final_z, avg_speed