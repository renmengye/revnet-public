"""Unit tests for multi-pass model."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import numpy as np
import tensorflow as tf
from resnet.configs import get_config
from resnet.configs import test_configs
from resnet.models.model_factory import get_model, get_multi_gpu_model
from resnet.models.multi_pass_model import MultiPassModel
from resnet.models.resnet_model import ResNetModel
from resnet.utils import logger

log = logger.get()
FOLDER = "tmp"
CKPT_FNAME = os.path.join(FOLDER, "test_multi_pass.ckpt")


class MultiPassModelTests(tf.test.TestCase):

  def _test_single_pass(self, method):
    config = get_config("resnet-test")
    config.momentum = 0.0
    config.base_learn_rate = 1e-1
    np.random.seed(0)
    BSIZE = config.batch_size
    xval = np.random.uniform(
        -1.0, 1.0, [BSIZE, config.height, config.width,
                    config.num_channel]).astype(np.float32)
    yval = np.floor(np.random.uniform(0, 9.9, [BSIZE])).astype(np.int32)

    # Run multi tower version.
    with tf.Graph().as_default(), self.test_session(
    ) as sess, log.verbose_level(2):
      x = tf.constant(xval)
      y = tf.constant(yval)
      with tf.variable_scope("Model", reuse=None):
        m1 = get_multi_gpu_model(
            "resnet", config, num_replica=2, inp=x, label=y)
      sess.run(tf.global_variables_initializer())
      m1.assign_lr(sess, config.base_learn_rate)
      tvars = tf.trainable_variables()
      tvars_str = map(lambda x: x.name, tvars)

      saver = tf.train.Saver(tvars)
      if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)
      saver.save(sess, CKPT_FNAME)
      m1.train_step(sess)
      tvars_v1 = sess.run(tvars)
      tvars_d1 = dict(zip(tvars_str, tvars_v1))

    # Run MultiPassModel.
    with tf.Graph().as_default(), self.test_session(
    ) as sess, log.verbose_level(2):
      with tf.variable_scope("Model", reuse=True):
        m2 = MultiPassModel(
            config,
            ResNetModel,
            num_passes=2,
            debug=True,
            inp=x,
            label=y,
            aggregate_method=method)
      tvars = tf.trainable_variables()
      saver = tf.train.Saver(tvars)
      sess.run(tf.global_variables_initializer())
      saver.restore(sess, CKPT_FNAME)
      m2.assign_lr(sess, config.base_learn_rate)
      m2.train_step(sess)
      tvars_v2 = sess.run(tvars)
      tvars_d2 = dict(zip(tvars_str, tvars_v2))

    for vv in tvars_str:
      log.info(vv, verbose=2)
      np.testing.assert_allclose(
          tvars_d1[vv], tvars_d2[vv], rtol=1e-4, atol=1e-6)
      log.info("...ok", verbose=2)

  def test_single_pass_cumsum(self):
    """Tests multi-pass is the same with multi-tower, single train step,
    using cumsum method."""
    self._test_single_pass("cumsum")

  def test_single_pass_storage(self):
    """Tests multi-pass is the same with multi-tower, single train step,
    using storage method."""
    self._test_single_pass("storage")

  def _test_multi_pass(self, method):
    config = get_config("resnet-test")
    config.momentum = 0.0
    config.base_learn_rate = 1e-1
    np.random.seed(0)
    BSIZE = config.batch_size
    xval = np.random.uniform(
        -1.0, 1.0, [BSIZE, config.height, config.width,
                    config.num_channel]).astype(np.float32)
    yval = np.floor(np.random.uniform(0, 9.9, [BSIZE])).astype(np.int32)

    # Run multi tower version.
    with tf.Graph().as_default(), self.test_session(
    ) as sess, log.verbose_level(2):
      x = tf.constant(xval)
      y = tf.constant(yval)
      with tf.variable_scope("Model", reuse=None):
        m1 = get_multi_gpu_model(
            "resnet", config, num_replica=2, inp=x, label=y)
      sess.run(tf.global_variables_initializer())
      m1.assign_lr(sess, config.base_learn_rate)
      tvars = tf.trainable_variables()
      tvars_str = map(lambda x: x.name, tvars)

      saver = tf.train.Saver(tvars)
      if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)
      saver.save(sess, CKPT_FNAME)
      for ii in range(3):
        m1.train_step(sess)
      tvars_v1 = sess.run(tvars)
      tvars_d1 = dict(zip(tvars_str, tvars_v1))

    # Run MultiPassModel.
    with tf.Graph().as_default(), self.test_session(
    ) as sess, log.verbose_level(2):
      with tf.variable_scope("Model", reuse=True):
        m2 = MultiPassModel(
            config,
            ResNetModel,
            num_passes=2,
            inp=x,
            label=y,
            aggregate_method=method)
      tvars = tf.trainable_variables()
      saver = tf.train.Saver(tvars)
      sess.run(tf.global_variables_initializer())
      m2.assign_lr(sess, config.base_learn_rate)
      saver.restore(sess, CKPT_FNAME)
      for ii in range(3):
        m2.train_step(sess)
      tvars_v2 = sess.run(tvars)
      tvars_d2 = dict(zip(tvars_str, tvars_v2))

    for vv in tvars_str:
      np.testing.assert_allclose(
          tvars_d1[vv], tvars_d2[vv], rtol=1e-4, atol=1e-6)

  def test_multi_pass_cumsum(self):
    """Tests multi-pass is the same with multi-tower, multi train step,
    using cumsum method."""
    self._test_multi_pass("cumsum")

  def test_multi_pass_storage(self):
    """Tests multi-pass is the same with multi-tower, multi train step,
    using storage method."""
    self._test_multi_pass("storage")


if __name__ == "__main__":
  tf.test.main()
