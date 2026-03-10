import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ده بيخفي كل الـ Warnings بتاعة TensorFlow
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # بيخلي بس الـ Errors هي اللي تظهر



# -*- coding: utf-8 -*-

import logging
import os
import pickle
import sys
import time
import warnings
from argparse import ArgumentParser
from pprint import pformat, pprint
import pandas as pd
import numpy as np
import tensorflow as tf
from tfsnippet.examples.utils import MLResults, print_with_title
from tfsnippet.scaffold import VariableSaver
from tfsnippet.utils import get_variables_as_dict, register_config_arguments, Config

from omni_anomaly.eval_methods import pot_eval, bf_search
from omni_anomaly.model import OmniAnomaly
from omni_anomaly.prediction import Predictor
from omni_anomaly.training import Trainer
from omni_anomaly.utils import get_data_dim, get_data, save_z


class ExpConfig(Config):
    # 🌍 Dataset بصمة: RobotArm
    dataset = "RobotArm"
    x_dim = get_data_dim(dataset)

    # 🧠 Model Architecture (الربط الذكي)
    use_connected_z_q = True
    use_connected_z_p = True

    # 🛠️ Parameters الموازنة للدراع:
    z_dim = 32  # [coreX Opt] Increased to 32 to capture dense correlations of 68 sensors
    rnn_cell = 'GRU' 
    rnn_num_hidden = 128
    window_length = 50 
    dense_dim = 128

    # 🌊 الـ Optimization "التقيل":
    posterior_flow_type = 'nf'  # تفعيل الـ Normalizing Flow
    nf_layers = 20              # دقة عالية جداً في رسم الـ Latent Space
    l2_reg = 0.0001             # منع الـ Overfitting (السر في دقة دراع الروبوت)
    std_epsilon = 1e-4          # هامش الأمان الرياضي

    # 🚀 بصمة coreX: (التعديلات اللي هتفرق في النتائج)
    # 1. الـ Root Cause Analysis (XAI)
    get_score_on_dim = True    # تفعليه فوراً عشان نعرف انهي موتور أو سنسور في الدراع هو اللي هيعطل
    
    # 2. الـ Dynamic Sensitivity (اختياري بس بيفرق)
    # ممكن تضيفي weights للسنسورات لو عارفة ان فيه سنسور أهم من التاني
    # sensor_weights = [1.2, 1.0, 0.8, ...] 

    # 3. الـ Advanced Evaluation
    test_n_z = 10              # زودنا الـ samples لـ 10 عشان الـ RCA يكون مستقر (Stable)
    level = 0.01              # [coreX Opt] Adjusted to 0.01 for balanced Precision/Recall
    
    # 📉 Training Logic
    max_epoch = 50
    batch_size = 25 
    initial_lr = 0.001
    early_stop = True

    # 📊 Visualization & Debugging (بصمتك في المناقشة)
    save_z = True              # لازم True عشان توري الدكتور الـ Clusters قبل وبعد الـ NF
    save_dir = 'model_coreX_v1' # تمييز الفولدر عشان المقارنات

    result_dir = os.path.join('results', 'RobotArm_coreX_v1')
    restore_dir = None  # Disabled for fresh training





def main():
    logging.basicConfig(
        level='INFO',
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    # 1. تحميل الداتا (coreX Loader)
    (x_train, _), (x_test, y_test) = get_data(dataset=config.dataset)    
    
    # تأمين أبعاد السنسورات
    if x_train.shape[1] > config.x_dim:
        print(f"--- [coreX Fix]: Clipping sensors from {x_train.shape[1]} to {config.x_dim} ---")
        x_train = x_train[:, :config.x_dim]
        x_test = x_test[:, :config.x_dim]

    # 2. بناء الـ Graph والـ Placeholders
    input_x = tf.placeholder(tf.float32, shape=[None, config.window_length, config.x_dim], name='input_x')
    
    with tf.variable_scope('model') as model_vs:
        model = OmniAnomaly(config=config, name="model")
        
        # تظبيط الباراميترز الاحتياطية
        config.test_batch_size = getattr(config, 'test_batch_size', config.batch_size)
        config.lr_anneal_epoch_freq = getattr(config, 'lr_anneal_epoch_freq', 10)
        config.lr_anneal_factor = getattr(config, 'lr_anneal_factor', 0.75)

        loss = model.get_training_loss(input_x)
        r_prob, z_info = model.get_score(input_x, n_z=config.test_n_z)

        trainer = Trainer(
            model=model, model_vs=model_vs,
            max_epoch=config.max_epoch, batch_size=config.batch_size,
            valid_batch_size=config.test_batch_size, initial_lr=config.initial_lr,
            lr_anneal_epochs=config.lr_anneal_epoch_freq, lr_anneal_factor=config.lr_anneal_factor,
            grad_clip_norm=config.gradient_clip_norm, valid_step_freq=config.valid_step_freq
        )

        predictor = Predictor(
            model, batch_size=config.batch_size, n_z=config.test_n_z,
            last_point_only=True
        )

    # 3. الـ Session و الـ Restore
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True

    with tf.Session(config=config_proto) as sess:
        sess.as_default()
        is_restored = False
        
        if config.restore_dir:
            checkpoint_path = os.path.abspath(config.restore_dir)
            latest_ckpt = tf.train.latest_checkpoint(checkpoint_path)
            if not latest_ckpt and "model.ckpt" in checkpoint_path:
                latest_ckpt = checkpoint_path

            if latest_ckpt:
                print(f"--- [coreX Info]: Found Checkpoint: {latest_ckpt} ---")
                saver = tf.train.Saver()
                try:
                    saver.restore(sess, latest_ckpt)
                    print("--- [coreX Info]: Success! Model Restored. ---")
                    is_restored = True
                except Exception as e:
                    print(f"--- [coreX Error]: Restore failed: {e} ---")

        # 4. التدريب (بيحصل لو مفيش Restore أو لو إنتي عايزة تزودي epochs)
        best_valid_metrics = {}
        if not is_restored:
            print("--- [coreX Info]: Starting Training Phase ---")
            sess.run(tf.global_variables_initializer())
            
            if config.max_epoch > 0:
                train_start = time.time()
                best_valid_metrics = trainer.fit(x_train)
                train_time = (time.time() - train_start) / max(1, config.max_epoch)
                best_valid_metrics['train_time_per_epoch'] = train_time
            else:
                print("--- [coreX Warning]: No checkpoint & max_epoch=0. Result might be poor. ---")
        else:
            print("--- [coreX Info]: Skipping Training, jumping to Evaluation ---")

        # 5. حساب الـ Scores (التنبؤ)
        print("--- [coreX Info]: Calculating Scores for Robot Arm ---")
        pred_start = time.time()
        train_score, train_z = predictor.get_score(x_train)[:2]
        test_score, test_z = predictor.get_score(x_test)[:2]
        pred_speed = (time.time() - pred_start) / (len(x_train) + len(x_test))

        # 6. التقييم وحل مشكلة الـ Labels (F1 & POT)
        if y_test is not None:
            # توحيد الـ Scores لـ Array واحدة
            test_score_final = np.sum(test_score, axis=-1) if test_score.ndim > 1 else test_score
            train_score_final = np.sum(train_score, axis=-1) if train_score.ndim > 1 else train_score
            
            # تنظيف وقص الـ Labels لتناسب الـ Windowing
            y_test_raw = np.array(y_test[-len(test_score_final):]).flatten()
            y_test_numeric = pd.to_numeric(y_test_raw, errors='coerce')
            y_test_cut = np.nan_to_num(y_test_numeric).astype(np.float32)
            y_test_cut = (y_test_cut > 0).astype(np.float32) # تحويل لـ Binary

            print(f"--- [coreX Audit]: Labels Cleaned. Unique: {np.unique(y_test_cut)} ---")

            # البحث عن أفضل F1
            t, th = bf_search(test_score_final, y_test_cut,
                              start=config.bf_search_min, end=config.bf_search_max,
                              step_num=100, display_freq=50)
            
            # حساب POT
            try:
                pot_result = pot_eval(train_score_final, test_score_final, y_test_cut, level=config.level)
            except:
                pot_result = {}

            # تجميع النتائج
            best_valid_metrics.update({
                'best-f1': t[0], 'precision': t[1], 'recall': t[2],
                'threshold': th, 'pred_speed_per_point': pred_speed
            })
            best_valid_metrics.update(pot_result)

        # 7. الحفظ النهائي (Results & Model)
        if not os.path.exists(config.result_dir): os.makedirs(config.result_dir)
        
        with open(os.path.join(config.result_dir, config.test_score_filename), 'wb') as f:
            pickle.dump(test_score, f)

        if not is_restored and config.save_dir:
            if not os.path.exists(config.save_dir): os.makedirs(config.save_dir)
            var_dict = get_variables_as_dict(model_vs)
            VariableSaver(var_dict, config.save_dir).save()
            print(f"--- [coreX Info]: Model saved to {config.save_dir}")

        # 8. التقرير النهائي
        print('\n' + '=' * 25 + ' coreX Final Report ' + '=' * 25)
        pprint(best_valid_metrics)
        print('=' * 70 + '\n')

        
                




    
if __name__ == '__main__':
    # 1. استدعاء الـ Config اللي بصمنا عليه (ExpConfig)
    config = ExpConfig()
    # --- سد ثغرات الـ Config عشان الكود ميفصلش ---
# القيم دي افتراضية (Standard) لنموذج OmniAnomaly
# --- سد ثغرات الـ Config النهائية ---


    missing_configs = {
            'test_batch_size': config.batch_size,
            'lr_anneal_factor': 0.75,
            'gradient_clip_norm': 5.0,
            'valid_step_freq': 100,
            'restore_dir': None,
            'lr_anneal_epoch_freq': 10,
            'max_epoch': 50,           # 50 epochs for production
            'test_n_z': 10,
            'test_batch_size': 25,
            'bf_search_min': -100,       # مهمين عشان الـ F1-Score ميفصلش
            'bf_search_max': 100,
            'bf_search_step_size': .1,
            'train_score_filename': 'train_score.pkl',
            'test_score_filename': 'test_score.pkl'
        }

    # عدلي السطر ده بالمسار اللي لسه جايباه
    # ضيفي /model.ckpt في آخر المسار
    # config.restore_dir = 'D:/Omni_Anomaly_Detection_coreX/results/RobotArm/20260222-172015/model.ckpt'

    for attr, value in missing_configs.items():
        if not hasattr(config, attr):
            setattr(config, attr, value)
    # --------------------------------------------

    # 2. معالجة الـ Arguments (لو حابة تغيري حاجة من الـ Terminal)
    arg_parser = ArgumentParser()
    register_config_arguments(config, arg_parser)
    arg_parser.parse_args(sys.argv[1:])
    
    # تركة coreX: التأكد إن الـ x_dim اتحدثت بناءً على الـ Dataset المختارة
    config.x_dim = get_data_dim(config.dataset)

    # 3. عرض الإعدادات (عشان تراجعي عليها قبل ما الـ GPU يسخن)
    print_with_title('Configurations', pformat(config.to_dict()), after='\n')

    # 4. تجهيز ملفات النتائج والموديلات (The Workspace)
    # تريكة: results.make_dirs بتضمن إن الفولدرات موجودة ومسحش القديم لو مش عايزة
    results = MLResults(config.result_dir)
    results.save_config(config)  # بنسيف الـ Config عشان نفضل فاكرين رنينا بـ Parameters إيه
    
    if config.save_dir:
        results.make_dirs(config.save_dir, exist_ok=True)
    
    # 5. الانطلاق (مع تجاهل صداع التحذيرات)
    # التركة دي مهمة عشان الـ TensorFlow 1.x و NumPy ساعات بيبقوا مش طايقين بعض
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module='numpy')
        warnings.filterwarnings("ignore", category=UserWarning) # زيادة أمان
        
        print(f"--- [coreX Info]: Launching OmniAnomaly for {config.dataset} ---")

        main()