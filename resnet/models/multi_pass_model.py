"""Multi-pass multi-tower CNN for training parallel GPU jobs."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import numpy as np
import tensorflow as tf

from resnet.models.multi_pass_optimizer import MultiPassOptimizer
from resnet.utils import logger

log = logger.get()


class MultiPassModel(object):
  """This model can average gradients from serial forward-backward propagations.
  """

  def __init__(self,
               config,
               model_cls,
               is_training=True,
               num_passes=2,
               inp=None,
               label=None,
               batch_size=None,
               aggregate_method="cumsum",
               debug=False):
    self._config = config
    self._model_cls = model_cls
    self._aggregate_method = aggregate_method
    self._debug = debug
    # Input.
    if inp is None:
      x = tf.placeholder(
          self.dtype,
          [batch_size, config.height, config.width, config.num_channel],
          name="x")
    else:
      x = inp
    if label is None:
      y = tf.placeholder(tf.int32, [batch_size], name="y")
    else:
      y = label

    self._pass_id = tf.placeholder(tf.int32, [], name="pass_id")
    self._input = x
    self._label = y
    self._model = None
    self._is_training = is_training
    self._num_passes = num_passes
    self._train_op_list = []
    self._batch_size = batch_size
    self._build_inference()
    self._build_optimizer()

  @property
  def dtype(self):
    tensor_type = os.getenv("TF_DTYPE", "float32")
    if tensor_type == "float32":
      return tf.float32
    else:
      return tf.float64

  @property
  def model(self):
    return self._model

  @property
  def train_op_list(self):
    return self._train_op_list

  def _build_inference(self):
    inp, label = self._slice_inp(self.input, self.label, self._pass_id)
    self._model = self._model_cls(
        self.config,
        is_training=self.is_training,
        inference_only=False,
        inp=inp,
        label=label,
        batch_size=self._batch_size,
        apply_grad=False)

  def _slice_inp(self, inp, label, idx):
    batch_size = tf.cast(tf.shape(inp)[0], tf.int32)
    num_per_pass = tf.cast(batch_size / self._num_passes, tf.int32)
    start = idx * num_per_pass
    return tf.slice(inp, [start, 0, 0, 0],
                    [num_per_pass, -1, -1, -1]), tf.slice(label, [start],
                                                          [num_per_pass])

  def _build_optimizer(self):
    config = self.config
    if not self.is_training:
      return

    self._lr = tf.get_variable(
        "learn_rate", [],
        initializer=tf.constant_initializer(0.0),
        dtype=self.dtype,
        trainable=False)
    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)
    opt = tf.train.MomentumOptimizer(self.lr, momentum=config.momentum)
    opt = MultiPassOptimizer(
        opt,
        num_passes=self.num_passes,
        debug=self._debug,
        aggregate_method=self._aggregate_method)
    self._optimizer = opt

    tf.get_variable_scope()._reuse = None
    global_step = tf.get_variable(
        "global_step", [],
        initializer=tf.constant_initializer(0.0),
        trainable=False,
        dtype=self.dtype)
    self._global_step = global_step

    # Add all trainable variables to the variable list.
    for ii in range(self.num_passes):
      self._train_op_list.append(
          opt.apply_gradients(
              self.model.grads_and_vars, global_step=global_step))

  def _slice_data(self, data, idx):
    # log.error(self.num_passes)
    num_per_pass = int(np.ceil(data.shape[0] / self.num_passes))
    # log.error(num_per_pass)
    start = idx * num_per_pass
    end = min(start + num_per_pass, data.shape[0])
    return data[start:end]

  def assign_lr(self, session, lr_value):
    """Assigns new learning rate."""
    log.info("Adjusting learning rate to {}".format(lr_value))
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  def train_step(self, sess, inp=None, label=None):
    """Run training."""
    ce = 0.0
    for ii, train_op in enumerate(self.train_op_list):
      if inp is not None:
        feed_data = {
            self.input: self._slice_data(inp, ii),
            self.label: self._slice_data(label, ii)
        }
      else:
        feed_data = dict()
      feed_data[self._pass_id] = ii
      results = sess.run(
          [self.model.cross_ent, train_op, self.model.bn_update_ops],
          feed_dict=feed_data)
      ce += results[0] / self.num_passes
    return ce

  def infer_step(self, sess, inp=None):
    """Run inference."""
    if inp is None:
      _feed_data = {self.model.input: inp}
    return sess.run(self.model.output, feed_dict=_feed_data)

  @property
  def input(self):
    return self._input

  @property
  def label(self):
    return self._label

  @property
  def output(self):
    return self._model.output

  @property
  def cost(self):
    return self._model.cost

  @property
  def cross_ent(self):
    return self._model.cross_ent

  @property
  def optimizer(self):
    return self._optimizer

  @property
  def num_passes(self):
    return self._num_passes

  @property
  def global_step(self):
    return self._global_step

  @property
  def config(self):
    return self._config

  @property
  def is_training(self):
    return self._is_training

  @property
  def num_replica(self):
    return self._num_replica

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op
