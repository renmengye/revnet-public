from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from resnet.configs import test_configs
from resnet.configs import get_config
from resnet.models.model_factory import get_model
from resnet.models.revnet_model import RevNetModel
from resnet.utils.test_utils import cosine_angle, get_degree, check_two_dict
from resnet.utils import logger

log = logger.get()


class RevNetModelTests(tf.test.TestCase):

  def _test_getmodel(self, modelname):
    with tf.Graph().as_default(), self.test_session(), log.verbose_level(2):
      config = get_config(modelname)
      get_model(config.model_class, config)

  def test_getmodel(self):
    self._test_getmodel("revnet-test")

  def test_getmodel_btl(self):
    self._test_getmodel("revnet-btl-test")

  def _test_grad(self, modelname):
    """Tests the manual gradients for every layer."""
    with tf.Graph().as_default(), self.test_session(
    ) as sess, log.verbose_level(2):
      config = get_config(modelname)
      config.manual_gradients = False
      config.filter_initialization = "uniform"

      # Declare a regular bprop model.
      m1 = get_model(config.model_class, config, is_training=True)
      name_list1 = map(lambda x: x[1].name, m1.grads_and_vars)
      grads1 = map(lambda x: x[0], m1.grads_and_vars)
      tf.get_variable_scope().reuse_variables()

      # Declare a manual bprop model.
      config.manual_gradients = True
      m2 = get_model(config.model_class, config, is_training=True)
      name_list2 = map(lambda x: x[1].name, m2.grads_and_vars)
      grads2 = map(lambda x: x[0], m2.grads_and_vars)

      # Check lengths are equal.
      self.assertEqual(len(m1.grads_and_vars), len(m2.grads_and_vars))

      # Prepare synthetic data.
      xval = np.random.uniform(-1.0, 1.0, [
          config.batch_size, config.height, config.width, config.num_channel
      ]).astype(np.float32)
      yval = np.floor(
          np.random.uniform(0.0, config.num_classes - 0.1, [config.batch_size
                                                           ])).astype(np.int32)
      sess.run(tf.global_variables_initializer())

      g1 = sess.run(grads1, feed_dict={m1.input: xval, m1.label: yval})
      gdict1 = dict(zip(name_list1, g1))
      g2 = sess.run(grads2, feed_dict={m2.input: xval, m2.label: yval})
      gdict2 = dict(zip(name_list2, g2))

      # Check two gradients are the same.
      check_two_dict(gdict1, gdict2, tol=1e0, name=modelname)

  def test_grad(self):
    self._test_grad("revnet-test")

  def test_grad_btl(self):
    self._test_grad("revnet-btl-test")


if __name__ == "__main__":
  tf.test.main()
