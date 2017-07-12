"""Unit tests for multi-tower model."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf
from resnet.configs import get_config
from resnet.configs import test_configs
from resnet.models import get_model, get_multi_gpu_model
from resnet.utils import logger
from resnet.utils.test_utils import check_two_dict

log = logger.get()


class MultiTowerModelTests(tf.test.TestCase):

  def test_fw(self):
    """Tests the forward computation is the same."""
    with tf.Graph().as_default(), self.test_session(
    ) as sess, log.verbose_level(2):
      config = get_config("resnet-test")
      config.num_channel = 4
      config.height = 8
      config.width = 8
      np.random.seed(0)
      xval = np.random.uniform(-1.0, 1.0, [10, 8, 8, 4]).astype(np.float32)
      x = tf.constant(xval)
      x1 = x[:5, :, :, :]
      x2 = x[5:, :, :, :]
      # We need to split two regular runs because of the complication brought by
      # batch normalization.
      with tf.variable_scope("Model", reuse=None):
        m11 = get_model("resnet", config, inp=x1)
      with tf.variable_scope("Model", reuse=True):
        m12 = get_model("resnet", config, inp=x2)
      with tf.variable_scope("Model", reuse=True):
        m2 = get_multi_gpu_model(
            "resnet", config, num_replica=2, inp=x)
      sess.run(tf.global_variables_initializer())
      y11, y12, y2 = sess.run([m11.output, m12.output, m2.output])
      np.testing.assert_allclose(y11, y2[:5, :], rtol=1e-5)
      np.testing.assert_allclose(y12, y2[5:, :], rtol=1e-5)

  def test_bk(self):
    """Tests the backward computation is the same."""
    with tf.Graph().as_default(), self.test_session(
    ) as sess, log.verbose_level(2):
      config = get_config("resnet-test")
      config.num_channel = 4
      config.height = 8
      config.width = 8
      np.random.seed(0)
      xval = np.random.uniform(-1.0, 1.0, [10, 8, 8, 4]).astype(np.float32)
      yval = np.floor(np.random.uniform(0, 9.9, [10])).astype(np.int32)
      x = tf.constant(xval)
      y = tf.constant(yval)
      # log.fatal(y.get_shape())
      x1 = x[:5, :, :, :]
      x2 = x[5:, :, :, :]
      y1 = y[:5]
      y2 = y[5:]
      with tf.variable_scope("Model", reuse=None):
        m11 = get_model("resnet", config, inp=x1, label=y1)
      with tf.variable_scope("Model", reuse=True):
        m12 = get_model("resnet", config, inp=x2, label=y2)
      with tf.variable_scope("Model", reuse=True):
        m2 = get_multi_gpu_model(
            "resnet", config, num_replica=2, inp=x, label=y)
      sess.run(tf.global_variables_initializer())
      tvars = tf.global_variables()

      name_list11 = map(lambda x: x[1].name, m11.grads_and_vars)
      grads11 = map(lambda x: x[0], m11.grads_and_vars)
      g11 = sess.run(grads11)
      gdict11 = dict(zip(name_list11, g11))
      name_list12 = map(lambda x: x[1].name, m12.grads_and_vars)
      grads12 = map(lambda x: x[0], m12.grads_and_vars)
      g12 = sess.run(grads12)
      gdict12 = dict(zip(name_list12, g12))

      name_list21 = map(lambda x: x[1].name, m2.tower_grads_and_vars[0])
      grads21 = map(lambda x: x[0], m2.tower_grads_and_vars[0])
      g21 = sess.run(grads21)
      gdict21 = dict(zip(name_list21, g21))
      name_list22 = map(lambda x: x[1].name, m2.tower_grads_and_vars[1])
      grads22 = map(lambda x: x[0], m2.tower_grads_and_vars[1])
      g22 = sess.run(grads22)
      gdict22 = dict(zip(name_list22, g22))

      # Check two gradients are the same.
      check_two_dict(gdict11, gdict21)
      check_two_dict(gdict12, gdict22)

      # Check the average gradients are the same.
      name_list2 = map(lambda x: x[1].name, m2.grads_and_vars)
      grads2 = map(lambda x: x[0], m2.grads_and_vars)
      g2 = sess.run(grads2)
      gdict2 = dict(zip(name_list2, g2))

      name_list1 = name_list11
      g1 = [(gdict11[kk] + gdict12[kk]) / 2.0 for kk in name_list1]
      gdict1 = dict(zip(name_list1, g1))
      check_two_dict(gdict1, gdict2)


if __name__ == "__main__":
  tf.test.main()
