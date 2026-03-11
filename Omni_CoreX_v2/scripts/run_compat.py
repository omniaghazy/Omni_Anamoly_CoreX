import sys
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
sys.modules['tensorflow'] = tf
sys.modules['tf'] = tf
sys.modules['tensorflow'].layers = tf.compat.v1.layers
sys.modules['tf'].layers = tf.compat.v1.layers

class MockContrib(object): pass
class MockFramework(object):
    def add_arg_scope(self, func): return func
    def arg_scope(self, *args, **kwargs):
        class DummyContextManager:
            def __enter__(self): pass
            def __exit__(self, exc_type, exc_val, exc_tb): pass
        return DummyContextManager()

tf.contrib = MockContrib()
tf.contrib.framework = MockFramework()
sys.modules['tensorflow.contrib'] = tf.contrib
sys.modules['tensorflow.contrib.framework'] = tf.contrib.framework
if not hasattr(tf, 'GraphKeys'):
    tf.GraphKeys = tf.compat.v1.GraphKeys


# Re-import tfsnippet now that tf is patched globally
import tfsnippet
import test_graph

try:
    test_graph.test_model()
except Exception as e:
    import traceback
    traceback.print_exc()
