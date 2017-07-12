"""Unit tests for multi-tower model."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf
from resnet.configs import get_config
from resnet.configs import test_configs
from resnet.models import get_model, get_multi_gpu_model
from resnet.models.multi_pass_optimizer import MultiPassOptimizer
from resnet.utils import logger
from resnet.utils.test_utils import check_two_dict

log = logger.get()


class MultiPassOptimizerTests(tf.test.TestCase):

  def test_basic(self):
    """Tests multi pass optimizer basic behaviour."""
    for aggregate_method in ["cumsum", "storage"]:
      with tf.Graph().as_default(), tf.Session() as sess, log.verbose_level(2):
        opt = tf.train.GradientDescentOptimizer(0.1)
        mp_opt = MultiPassOptimizer(opt, 2, aggregate_method=aggregate_method)
        a = tf.get_variable(
            "a", shape=[10, 12], initializer=tf.constant_initializer(0.0))
        b = tf.get_variable(
            "b", shape=[11, 13], initializer=tf.constant_initializer(0.0))

        da1 = tf.ones([10, 12]) * 0.4
        da2 = tf.ones([10, 12]) * 0.6

        db1 = tf.ones([11, 13]) * 0.8
        db2 = tf.ones([11, 13]) * 1.0

        gv1 = [(da1, a), (db1, b)]
        gv2 = [(da2, a), (db2, b)]

        op1 = mp_opt.apply_gradients(gv1)
        op2 = mp_opt.apply_gradients(gv2)

        sess.run(tf.global_variables_initializer())
        sess.run([op1])
        sess.run([op2])
        a, b = sess.run([a, b])

        # Final value equals -learning_rate * average_gradients.
        np.testing.assert_allclose(a, -np.ones([10, 12]) * 0.05)
        np.testing.assert_allclose(b, -np.ones([11, 13]) * 0.09)

if __name__ == "__main__":
  tf.test.main()
