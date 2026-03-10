import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# -*- coding: utf-8 -*-
import logging
import pickle
import sys
import time
import warnings
from argparse import ArgumentParser
from pprint import pformat, pprint

import numpy as np
import pandas as pd
from tfsnippet.examples.utils import MLResults, print_with_title
from tfsnippet.scaffold import VariableSaver
from tfsnippet.utils import get_variables_as_dict, register_config_arguments, Config

from omni_anomaly.eval_methods import pot_eval, bf_search
from omni_anomaly.model import OmniAnomaly
from omni_anomaly.prediction import Predictor
from omni_anomaly.training import Trainer
from omni_anomaly.utils import get_data_dim, get_data, save_z


class ExpConfig(Config):
    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = "RobotArm"
    x_dim   = get_data_dim(dataset)   # auto-detected from PKL (e.g. 36)

    # ── Model Architecture ────────────────────────────────────────────────────
    use_connected_z_q = True
    use_connected_z_p = True

    # ── [OPTIMIZED] Upgraded capacity and temporal context ──────────────────
    z_dim          = 64    # [UPGRADED] 32→64: more latent capacity
    rnn_cell       = 'GRU'
    rnn_num_hidden = 256
    window_length  = 120   # [UPGRADED] 100→120: capture longer patterns
    dense_dim      = 256
    beta           = 0.5   # [NEW] beta-VAE weight (0.5 favors reconstruction)

    # ── Normalizing Flow ──────────────────────────────────────────────────────
    posterior_flow_type = 'nf'
    nf_layers           = 20
    l2_reg              = 0.0001
    std_epsilon         = 1e-4

    # ── Evaluation ────────────────────────────────────────────────────────────
    get_score_on_dim = True    # per-sensor RCA
    test_n_z         = 10      # Monte-Carlo samples for stable score
    level            = 0.01   # POT risk level

    # ── Training ──────────────────────────────────────────────────────────────
    max_epoch   = 200      # [UPGRADED] 100→200: allow for better convergence
    batch_size  = 64       # [ADJUSTED] 50→64: more stable gradients
    initial_lr  = 0.001
    early_stop  = True

    # ── Paths ─────────────────────────────────────────────────────────────────
    save_z       = True
    save_dir     = 'model_coreX_v2_optimized'
    result_dir   = os.path.join('results', 'RobotArm_coreX_v2_optimized')
    restore_dir  = None    # Train from scratch for optimization test


def main():
    logging.basicConfig(
        level='INFO',
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    # ── 1. Load Data ──────────────────────────────────────────────────────────
    (x_train, _), (x_test, y_test) = get_data(dataset=config.dataset)

    # Safety: clip to configured x_dim if data has more sensors
    if x_train.shape[1] != config.x_dim:
        dim = min(x_train.shape[1], config.x_dim)
        print(f"[coreX] Aligning sensor dim: data={x_train.shape[1]} → config={config.x_dim}, using {dim}")
        x_train = x_train[:, :dim]
        x_test  = x_test[:,  :dim]
        config.x_dim = dim

    print(f"[coreX] Train shape: {x_train.shape}  |  Test shape: {x_test.shape}")

    # ── 2. Build TF Graph ─────────────────────────────────────────────────────
    with tf.variable_scope('model') as model_vs:
        model = OmniAnomaly(config=config, name="model")

        # Fill optional config fields
        config.test_batch_size      = getattr(config, 'test_batch_size',      config.batch_size)
        config.lr_anneal_epoch_freq  = getattr(config, 'lr_anneal_epoch_freq',  10)
        config.lr_anneal_factor      = getattr(config, 'lr_anneal_factor',      0.75)
        config.gradient_clip_norm    = getattr(config, 'gradient_clip_norm',    5.0)
        config.valid_step_freq       = getattr(config, 'valid_step_freq',       100)

        # Use a dummy placeholder just for the training loss graph
        input_x = tf.placeholder(
            tf.float32,
            shape=[None, config.window_length, config.x_dim],
            name='input_x'
        )
        loss = model.get_training_loss(input_x)

        trainer = Trainer(
            model=model, model_vs=model_vs,
            max_epoch=config.max_epoch, batch_size=config.batch_size,
            valid_batch_size=config.test_batch_size,
            initial_lr=config.initial_lr,
            lr_anneal_epochs=config.lr_anneal_epoch_freq,
            lr_anneal_factor=config.lr_anneal_factor,
            grad_clip_norm=config.gradient_clip_norm,
            valid_step_freq=config.valid_step_freq
        )

        # [FIX] Predictor builds its own graph lazily — no extra input_x clash
        predictor = Predictor(
            model,
            batch_size=config.test_batch_size,
            n_z=config.test_n_z,
            last_point_only=True
        )

    # ── 3. Session & Restore ─────────────────────────────────────────────────
    cfg_proto = tf.ConfigProto()
    cfg_proto.gpu_options.allow_growth = True

    with tf.Session(config=cfg_proto) as sess:
        sess.as_default()
        is_restored = False

        if config.restore_dir:
            ckpt_path = os.path.abspath(config.restore_dir)
            latest   = tf.train.latest_checkpoint(ckpt_path)
            if latest:
                print(f"[coreX] Restoring checkpoint: {latest}")
                saver = tf.train.Saver()
                try:
                    saver.restore(sess, latest)
                    print("[coreX] ✅ Model restored successfully.")
                    is_restored = True
                except Exception as e:
                    print(f"[coreX] ❌ Restore failed: {e}")

        # ── 4. Training (skipped if restored) ────────────────────────────────
        best_valid_metrics = {}
        if not is_restored:
            print("[coreX] Starting Training Phase...")
            sess.run(tf.global_variables_initializer())

            if config.max_epoch > 0:
                t0 = time.time()
                best_valid_metrics = trainer.fit(x_train)
                elapsed = (time.time() - t0) / max(1, config.max_epoch)
                best_valid_metrics['train_time_per_epoch'] = elapsed
        else:
            print("[coreX] ⏩ Skipping training — jumping to evaluation.")

        # ── 5. Score Both Splits ─────────────────────────────────────────────
        print("\n[coreX] 📊 Computing anomaly scores...")
        t0 = time.time()
        train_score, train_z, _ = predictor.get_score(x_train)
        test_score,  test_z,  _ = predictor.get_score(x_test)
        pred_elapsed = time.time() - t0
        print(f"[coreX] Scoring done in {pred_elapsed:.1f}s  "
              f"| train_score: {train_score.shape}  test_score: {test_score.shape}")

        # ── 6. Evaluate ───────────────────────────────────────────────────────
        if y_test is not None:
            # Scores are already 1-D (Predictor collapses per-sensor)
            test_score_1d  = test_score.flatten()
            train_score_1d = train_score.flatten()

            # [FIX] Align labels to windowed score length (take tail)
            y_raw = np.array(y_test).flatten()
            y_raw = pd.to_numeric(y_raw, errors='coerce')
            y_bin = np.nan_to_num(y_raw).astype(np.float32)
            y_bin = (y_bin > 0).astype(np.float32)
            y_aligned = y_bin[-len(test_score_1d):]

            print(f"\n[coreX] Label audit — unique: {np.unique(y_aligned)}  "
                  f"anomaly pts: {int(y_aligned.sum())} / {len(y_aligned)}")

            # [IMPROVEMENT] BF-Search over actual score range (not fixed -100→100)
            score_min = float(test_score_1d.min())
            score_max = float(test_score_1d.max())
            margin    = (score_max - score_min) * 0.05   # 5% padding
            bf_start  = score_min - margin
            bf_end    = score_max + margin

            print(f"[coreX] BF-Search range: [{bf_start:.2f}, {bf_end:.2f}]")
            t_metrics, best_th = bf_search(
                test_score_1d, y_aligned,
                start=bf_start, end=bf_end,
                step_num=500, display_freq=100
            )

            # POT evaluation
            try:
                pot_result = pot_eval(
                    train_score_1d, test_score_1d, y_aligned,
                    level=config.level
                )
            except Exception as e:
                print(f"[coreX] POT failed: {e}")
                pot_result = {}

            # Assemble final metrics
            best_valid_metrics.update({
                'best-f1':     t_metrics[0],
                'precision':   t_metrics[1],
                'recall':      t_metrics[2],
                'accuracy':    t_metrics[3],
                'bf-threshold': best_th,
                'pred_time_s':  pred_elapsed,
            })
            best_valid_metrics.update(pot_result)

        # ── 7. Save Results ───────────────────────────────────────────────────
        os.makedirs(config.result_dir, exist_ok=True)

        with open(os.path.join(config.result_dir, config.test_score_filename),  'wb') as f:
            pickle.dump(test_score, f)
        with open(os.path.join(config.result_dir, config.train_score_filename), 'wb') as f:
            pickle.dump(train_score, f)
        print(f"[coreX] Scores saved to {config.result_dir}")

        # Save model if we trained from scratch
        if not is_restored and config.save_dir:
            os.makedirs(config.save_dir, exist_ok=True)
            try:
                saver = tf.train.Saver(var_list=tf.global_variables())
                saver.save(sess, os.path.join(config.save_dir, 'model.ckpt'))
                print(f"[coreX] ✅ Model checkpoint saved to {config.save_dir}")
            except Exception as e:
                print(f"[coreX] ❌ Save failed: {e}")

        # ── 8. Final Report ───────────────────────────────────────────────────
        print('\n' + '=' * 60)
        print('          coreX OmniAnomaly - Final Report')
        print('=' * 60)
        pprint(best_valid_metrics)
        print('=' * 60 + '\n')


if __name__ == '__main__':
    config = ExpConfig()

    # Fill any missing config attributes with sensible defaults
    _defaults = {
        'test_batch_size':      config.batch_size,
        'lr_anneal_factor':     0.75,
        'gradient_clip_norm':   5.0,
        'valid_step_freq':      100,
        'lr_anneal_epoch_freq': 10,
        'max_epoch':            100,
        'test_n_z':             10,
        'bf_search_min':        -1e6,   # placeholder — auto-computed from data
        'bf_search_max':        0.0,    # placeholder
        'bf_search_step_size':  0.1,
        'train_score_filename': 'train_score.pkl',
        'test_score_filename':  'test_score.pkl',
    }
    for attr, val in _defaults.items():
        if not hasattr(config, attr):
            setattr(config, attr, val)

    # Parse CLI overrides
    arg_parser = ArgumentParser()
    register_config_arguments(config, arg_parser)
    arg_parser.parse_args(sys.argv[1:])

    # Re-detect x_dim in case --dataset was overridden via CLI
    config.x_dim = get_data_dim(config.dataset)

    print_with_title('Configurations', pformat(config.to_dict()), after='\n')

    results = MLResults(config.result_dir)
    results.save_config(config)
    if config.save_dir:
        results.make_dirs(config.save_dir, exist_ok=True)

    print(f"[coreX] Launching OmniAnomaly for dataset: {config.dataset}")
    print(f"[coreX] window_length={config.window_length} | "
          f"rnn_hidden={config.rnn_num_hidden} | z_dim={config.z_dim}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        main()