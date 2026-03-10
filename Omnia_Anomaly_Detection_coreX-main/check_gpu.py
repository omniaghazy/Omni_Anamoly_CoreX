import tensorflow as tf
import os

print("--- [coreX System Audit] ---")

# 1. بنشوف التنسرفلو شايف كام كارت شاشة
devices = tf.config.experimental.list_physical_devices('GPU') if hasattr(tf.config, 'experimental') else []
print(f"Num GPUs Available: {len(devices)}")

# 2. بنطبع تفاصيل الكروت لو موجودة
from tensorflow.python.client import device_lib
local_device_protos = device_lib.list_local_devices()
gpus = [x for x in local_device_protos if x.device_type == 'GPU']

if gpus:
    for i, gpu in enumerate(gpus):
        print(f"GPU {i}: {gpu.physical_device_desc}")
else:
    print("No NVIDIA GPU detected by TensorFlow. (Checking CPU instead...)")

# 3. اختبار عملي (نخلي التنسرفلو يعمل عملية حسابية على الـ GPU)
try:
    with tf.device('/gpu:0'):
        a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
        b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')
        c = a + b
        print("Success: TensorFlow can compute on GPU!")
except Exception as e:
    print(f"Failed to compute on GPU: {e}")