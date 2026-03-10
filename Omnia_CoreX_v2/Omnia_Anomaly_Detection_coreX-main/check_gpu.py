import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

print("--- [coreX System Audit] ---")
print(f"TensorFlow version: {tf.__version__}")

# 1. TF1-safe GPU detection using device_lib
from tensorflow.python.client import device_lib
local_devices = device_lib.list_local_devices()
gpus = [x for x in local_devices if x.device_type == 'GPU']

if gpus:
    print(f"[OK] {len(gpus)} GPU(s) detected:")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.physical_device_desc}")
else:
    print("[WARN] No GPU detected. Training will use CPU.")

# 2. tf.test check
gpu_name = tf.test.gpu_device_name()
if gpu_name:
    print(f"[OK] GPU device name: {gpu_name}")
else:
    print("[INFO] No GPU from tf.test.gpu_device_name() (CPU mode).")

# 3. Computation test using TF1 Session
try:
    with tf.Session() as sess:
        device = '/gpu:0' if gpus else '/cpu:0'
        with tf.device(device):
            a = tf.constant([1.0, 2.0, 3.0])
            b = tf.constant([1.0, 2.0, 3.0])
            result = sess.run(a + b)
    print(f"[OK] Computation test passed on {device}: {result}")
except Exception as e:
    print(f"[ERR] Computation test failed: {e}")