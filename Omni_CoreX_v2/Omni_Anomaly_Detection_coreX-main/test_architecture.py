
# -*- coding: utf-8 -*-
import os
import sys

# ─────────────────────────────────────────────────────────────────────────────
# 1. SHIM LAYER (Must be first to prevent tfsnippet/tf2 import crashes)
# ─────────────────────────────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 1.1 Force legacy Keras or shim layers directly to avoid "dense not available with Keras 3"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
if not hasattr(tf, 'layers') or 'keras' in str(tf.layers):
    tf.layers = tf.compat.v1.layers
    sys.modules['tensorflow.layers'] = tf.compat.v1.layers

class MockContrib(object):
    def __init__(self):
        self.rnn = self
        self.framework = MockFramework()
        self.layers = MockLayers()
    def static_bidirectional_rnn(self, *args, **kwargs):
        # outputs, fw_state, bw_state
        return [tf.zeros_like(args[2][0])] * len(args[2]), None, None

class MockLayers(object):
    def layer_norm(self, inputs, *args, **kwargs): return inputs

class MockFramework(object):
    def add_arg_scope(self, func): return func
    def arg_scope(self, *args, **kwargs):
        class DummyContextManager:
            def __enter__(self): pass
            def __exit__(self, exc_type, exc_val, exc_tb): pass
        return DummyContextManager()

mock_contrib = MockContrib()
tf.contrib = mock_contrib
sys.modules['tensorflow.contrib'] = mock_contrib
sys.modules['tensorflow.contrib.framework'] = mock_contrib.framework
sys.modules['tensorflow.contrib.layers'] = mock_contrib.layers
sys.modules['tensorflow.contrib.rnn'] = mock_contrib

# Standard math aliases for TF 2.x
if not hasattr(tf, 'log'): tf.log = tf.math.log
if not hasattr(tf, 'exp'): tf.exp = tf.math.exp
if not hasattr(tf, 'sqrt'): tf.sqrt = tf.math.sqrt

# Apply to the base 'tensorflow' module as well
import tensorflow as tf_base
tf_base.log = tf.math.log
tf_base.exp = tf.math.exp
tf_base.sqrt = tf.math.sqrt
sys.modules['tensorflow'].log = tf.math.log
sys.modules['tensorflow'].exp = tf.math.exp
sys.modules['tensorflow'].sqrt = tf.math.sqrt

# ─────────────────────────────────────────────────────────────────────────────
import numpy as np

# Mock ExpConfig to avoid main.py import complexities
class MockConfig:
    def __init__(self):
        self.window_length = 120
        self.x_dim = 15
        self.z_dim = 64
        self.rnn_cell = 'GRU'
        self.rnn_num_hidden = 256
        self.dense_dim = 256
        self.beta = 0.5
        self.posterior_flow_type = None
        self.nf_layers = 1
        self.l2_reg = 0.0001
        self.std_epsilon = 1e-4
        self.use_connected_z_q = True
        self.use_connected_z_p = True
        self.causal_adj_matrix_path = 'dummy_causal_test.npy'
        self.adm_layers = 3
        self.n_heads = 8
        self.k_weight = 3.0

    def to_dict(self): return vars(self)

# Now import OmniAnomaly (which imports tfsnippet)
from omni_anomaly.model import OmniAnomaly

def run_dry_run():
    print("="*80)
    print("      OmniAnomaly Hybrid Y-Split - DRY RUN VERIFICATION (STANDALONE)")
    print("="*80)

    config = MockConfig()
    
    # 1. Create dummy causal matrix
    if not os.path.exists(config.causal_adj_matrix_path):
        dummy_adj = np.eye(config.x_dim).astype(np.float32)
        np.save(config.causal_adj_matrix_path, dummy_adj)

    # 2. Build Graph
    tf.reset_default_graph()
    input_x = tf.placeholder(tf.float32, shape=[None, config.window_length, config.x_dim], name='input_x')
    
    try:
        with tf.variable_scope('model'):
            model = OmniAnomaly(config=config, name="model")
            
            # This triggers the _hybrid_encoder logic: GCN -> [RNN branch, AD branch] -> Concat
            input_q_tensor = model.vae.h_for_q_z(input_x)['input_q']
            
            # Validate loss graph
            loss = model.get_training_loss(input_x)

        print("\n✅ TF Graph Construction: SUCCESS")

        # 3. Execution
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            dummy_data = np.random.rand(16, 120, 15).astype(np.float32)
            out_features, out_loss = sess.run([input_q_tensor, loss], feed_dict={input_x: dummy_data})
            
            print("\nVERIFICATION RESULTS:")
            print(f"Input Shape:           {dummy_data.shape}")
            print(f"Concatenated Features: {out_features.shape}")
            print(f"Initial Multi-Loss:    {out_loss:.4f}")
            
            expected_feat_dim = config.rnn_num_hidden + config.dense_dim
            if out_features.shape[-1] == expected_feat_dim:
                print(f"\n[OK] SUCCESS: Concatenated dimension ({out_features.shape[-1]}) matches expected ({expected_feat_dim})")
            else:
                print(f"\n[ERROR] ERROR: Dimension mismatch! Got {out_features.shape[-1]}, expected {expected_feat_dim}")

    except Exception as e:
        print(f"\n[FAIL] CRITICAL FAILURE during graph construction/execution:")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    if os.path.exists(config.causal_adj_matrix_path):
        os.remove(config.causal_adj_matrix_path)

    print("\n" + "="*80)

if __name__ == "__main__":
    run_dry_run()
