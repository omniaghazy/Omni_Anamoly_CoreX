import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Omni_Anomaly_Detection_coreX-main'))
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Mock tf.contrib because TF2 removed it, but tfsnippet expects it
import sys
class MockContrib(object):
    def __init__(self):
        self.rnn = self
        self.framework = MockFramework()
        self.layers = MockLayers()
    def static_bidirectional_rnn(self, *args, **kwargs):
        # Return something that looks like rnn outputs
        # outputs, fw_state, bw_state
        return [tf.zeros_like(args[2][0])] * len(args[2]), None, None

class MockLayers(object):
    def layer_norm(self, inputs, *args, **kwargs):
        return inputs

class MockFramework(object):

    def add_arg_scope(self, func):
        return func
    def arg_scope(self, *args, **kwargs):
        class DummyContextManager:
            def __enter__(self): pass
            def __exit__(self, exc_type, exc_val, exc_tb): pass
        return DummyContextManager()

mock_contrib = MockContrib()
tf.contrib = mock_contrib
sys.modules['tensorflow.contrib'] = mock_contrib
sys.modules['tensorflow.contrib.framework'] = mock_contrib.framework
sys.modules['tensorflow.contrib.rnn'] = mock_contrib

if not hasattr(tf, 'log'): tf.log = tf.math.log
if not hasattr(tf, 'exp'): tf.exp = tf.math.exp
if not hasattr(tf, 'sqrt'): tf.sqrt = tf.math.sqrt
if not hasattr(tf, 'div'): tf.div = tf.math.divide
if not hasattr(tf, 'mod'): tf.mod = tf.math.mod
if not hasattr(tf, 'floor'): tf.floor = tf.math.floor
if not hasattr(tf, 'ceil'): tf.ceil = tf.math.ceil

if not hasattr(tf, 'GraphKeys'):
    tf.GraphKeys = tf.compat.v1.GraphKeys
if not hasattr(tf, 'layers'):
    tf.layers = tf.compat.v1.layers
sys.modules['tensorflow'].layers = tf.compat.v1.layers
sys.modules['tensorflow'].GraphKeys = tf.compat.v1.GraphKeys
sys.modules['tensorflow'].log = tf.math.log
sys.modules['tensorflow'].exp = tf.math.exp




import tfsnippet
from omni_anomaly.model import OmniAnomaly

class TestConfig:
    x_dim = 15
    z_dim = 64
    rnn_cell = 'GRU'
    rnn_num_hidden = 256
    window_length = 120
    dense_dim = 256
    beta = 0.5
    posterior_flow_type = None
    nf_layers = 1
    l2_reg = 0.0001
    std_epsilon = 1e-4
    use_connected_z_q = True
    use_connected_z_p = True
    causal_adj_matrix_path = 'dummy_adj_matrix.npy'

def test_model():
    print("1. Creating dummy configuration and matrix...")
    config = TestConfig()
    
    # Create fake causal adjacency matrix
    dummy_matrix = np.random.rand(15, 15)
    np.save('dummy_adj_matrix.npy', dummy_matrix)
    
    # Create dummy input tensor
    batch_size = 16
    window_length = config.window_length
    x_dim = config.x_dim
    
    print(f"   Input shape expected: [{batch_size}, {window_length}, {x_dim}]")
    
    print("2. Building TensorFlow graph...")
    tf.reset_default_graph()
    input_x = tf.placeholder(tf.float32, shape=[None, window_length, x_dim], name='input_x')
    
    with tf.variable_scope('model'):
        model = OmniAnomaly(config=config, name="model")
        
        # Test get_training_loss
        loss = model.get_training_loss(input_x)
        print("   [SUCCESS] get_training_loss() graph built successfully.")
        
        # Test get_score
        score, z_info = model.get_score(input_x, n_z=10, last_point_only=True)
        print("   [SUCCESS] get_score() graph built successfully.")

        
    print("3. Running session...")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        dummy_x = np.random.rand(batch_size, window_length, x_dim).astype(np.float32)
        
        # Run loss
        loss_val = sess.run(loss, feed_dict={input_x: dummy_x})
        print(f"   [SUCCESS] Loss calculated: {loss_val:.4f}")
        
        # Run score
        score_val, z_val = sess.run([score, z_info], feed_dict={input_x: dummy_x})
        print(f"   [SUCCESS] Score calculated. Shape: {score_val.shape} (Expected: ({batch_size}, {x_dim}))")
        print(f"   [SUCCESS] Z Info calculated. Shape: {z_val.shape}")
        
    print("4. Cleaning up...")
    if os.path.exists('dummy_adj_matrix.npy'):
        os.remove('dummy_adj_matrix.npy')

if __name__ == "__main__":
    test_model()
