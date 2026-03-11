import sys
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
sys.modules['tensorflow'] = tf
sys.modules['tf'] = tf

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

import traceback
try:
    from tfsnippet.distributions import Distribution
    print("SUCCESS")
except Exception as e:
    with open('real_trace.txt', 'w') as f:
        traceback.print_exc(file=f)
    print("TRACE WRITTEN")
