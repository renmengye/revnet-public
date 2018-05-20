# Author: Mengye Ren (mren@cs.toronto.edu).
#
# Modified from Tensorflow original code.
# Original Tensorflow license shown below.
# =============================================================================
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""ResNet model.

Related papers:
https://arxiv.org/pdf/1603.05027v2.pdf
https://arxiv.org/pdf/1512.03385v1.pdf
https://arxiv.org/pdf/1605.07146v1.pdf
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from resnet.models.nnlib import concat, weight_variable_cpu, batch_norm
from resnet.models.model_factory import RegisterModel
from resnet.utils import logger

log = logger.get()


@RegisterModel("resnet")
class ResnetModel(object):
  """ResNet model."""

  def __init__(self,
               config,
               is_training=True,
               inference_only=False,
               inp=None,
               label=None,
               dtype=tf.float32,
               batch_size=None,
               apply_grad=True,
               idx=0):
    """ResNet constructor.

    Args:
      config: Hyperparameters.
      is_training: One of "train" and "eval".
      inference_only: Do not build optimizer.
    """
    self._config = config
    self._dtype = dtype
    self._apply_grad = apply_grad
    self._saved_hidden = []
    # Debug purpose only.
    self._saved_hidden2 = []
    self._bn_update_ops = None
    self.is_training = is_training
    self._batch_size = batch_size
    self._dilated = False

    # Input.
    if inp is None:
      x = tf.placeholder(
          dtype, [batch_size, config.height, config.width,
                  config.num_channels], "x")
    else:
      x = inp

    if label is None:
      y = tf.placeholder(tf.int32, [batch_size], "y")
    else:
      y = label

    logits = self.build_inference_network(x)
    predictions = tf.nn.softmax(logits)

    with tf.variable_scope("costs"):
      xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=y)
      xent = tf.reduce_mean(xent, name="xent")
      cost = xent
      cost += self._decay()

    self._cost = cost
    self._input = x
    # Make sure that the labels are in reasonable range.
    # with tf.control_dependencies(
    #     [tf.assert_greater_equal(y, 0), tf.assert_less(y, config.num_classes)]):
    #   self._label = tf.identity(y)
    self._label = y
    self._cross_ent = xent
    self._output = predictions
    self._output_idx = tf.cast(tf.argmax(predictions, axis=1), tf.int32)
    self._correct = tf.to_float(
        tf.equal(tf.cast(self._output_idx, self.label.dtype), self.label))

    if not is_training or inference_only:
      return

    global_step = tf.get_variable(
        "global_step", [],
        initializer=tf.constant_initializer(0),
        trainable=False,
        dtype=tf.int64)
    lr = tf.train.piecewise_constant(
        global_step, self.config.learn_rate_decay_steps,
        [self.config.learn_rate] + list(self.config.learn_rate_list))
    self._lr = lr
    self._grads_and_vars = self._compute_gradients(cost)
    tf.get_variable_scope()._reuse = None
    if self._apply_grad:
      tf.get_variable_scope()._reuse = None
      self._train_op = self._apply_gradients(
          self._grads_and_vars, global_step=global_step, name="train_step")
    self._global_step = global_step

  def _apply_gradients(self,
                       grads_and_vars,
                       global_step=None,
                       name="train_step"):
    """Apply the gradients globally."""
    if self.config.optimizer_type == "sgd":
      opt = tf.train.GradientDescentOptimizer(self.lr)
    elif self.config.optimizer_type == "momentum":
      opt = tf.train.MomentumOptimizer(self.lr, 0.9)
    train_op = opt.apply_gradients(
        self._grads_and_vars, global_step=global_step, name="train_step")
    return train_op

  def _compute_gradients(self, cost, var_list=None):
    """Compute the gradients to variables."""
    if var_list is None:
      var_list = tf.trainable_variables()
    grads = tf.gradients(cost, var_list, gate_gradients=True)
    return zip(grads, var_list)

  def _init_conv(self, x, n_filters):
    """Build initial conv layers."""
    config = self.config
    init_filter_size = config.init_filter_size
    with tf.variable_scope("init"):
      h = self._conv("init_conv", x, init_filter_size, self.config.num_channels,
                     n_filters, self._stride_arr(config.init_stride))
      h = self._batch_norm("init_bn", h)
      h = self._relu("init_relu", h)
      # Max-pooling is used in ImageNet experiments to further reduce
      # dimensionality.
      if config.init_max_pool:
        h = tf.nn.max_pool(
            h,
            self._stride_arr(3),
            self._stride_arr(2),
            "SAME",
            data_format=self.config.data_format)
    return h

  def build_inference_network(self, x):
    config = self.config
    is_training = self.is_training
    num_stages = len(self.config.num_residual_units)
    strides = config.strides
    num_filters = [ff for ff in config.num_filters]  # Copy filter config.
    h = self._init_conv(x, num_filters[0])
    if config.use_bottleneck:
      res_func = self._bottleneck_residual
      # For CIFAR-10 it's [16, 16, 32, 64] => [16, 64, 128, 256]
      for ii in range(1, len(num_filters)):
        num_filters[ii] *= 4
    else:
      res_func = self._residual

    # New version, single for-loop. Easier for checkpoint.
    nlayers = sum(config.num_residual_units)
    ss = 0
    ii = 0
    for ll in range(nlayers):
      # Residual unit configuration.
      if ss == 0 and ii == 0:
        no_activation = True
      else:
        no_activation = False
      if ii == 0:
        if ss == 0:
          no_activation = True
        else:
          no_activation = False
        in_filter = num_filters[ss]
        stride = self._stride_arr(strides[ss])
      else:
        in_filter = num_filters[ss + 1]
        stride = self._stride_arr(1)
      out_filter = num_filters[ss + 1]

      # Save hidden state.
      if ii == 0:
        self._saved_hidden.append(h)

      # Build residual unit.
      with tf.variable_scope("unit_{}_{}".format(ss + 1, ii)):
        h = res_func(
            h, in_filter, out_filter, stride, no_activation=no_activation)

      if (ii + 1) % config.num_residual_units[ss] == 0:
        ss += 1
        ii = 0
      else:
        ii += 1

    # Save hidden state.
    self._saved_hidden.append(h)

    # Make a single tensor.
    if type(h) == tuple:
      h = concat(h, axis=3)

    with tf.variable_scope("unit_last"):
      h = self._batch_norm("final_bn", h)
      h = self._relu("final_relu", h)

    h = self._global_avg_pool(h)

    # Classification layer.
    with tf.variable_scope("logit"):
      logits = self._fully_connected(h, config.num_classes)

    return logits

  def _weight_variable(self,
                       shape,
                       init_method=None,
                       dtype=tf.float32,
                       init_param=None,
                       wd=None,
                       name=None,
                       trainable=True,
                       seed=0):
    """Wrapper to declare variables. Default on CPU."""
    return weight_variable_cpu(
        shape,
        init_method=init_method,
        dtype=dtype,
        init_param=init_param,
        wd=wd,
        name=name,
        trainable=trainable,
        seed=seed)

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    if self.config.data_format == "NHWC":
      return [1, stride, stride, 1]
    elif self.config.data_format == "NCHW":
      return [1, 1, stride, stride]
    else:
      raise ValueError("Unknown data format")

  def _batch_norm(self, name, x, add_ops=True):
    """Batch normalization."""
    bn = tf.contrib.layers.batch_norm(
        x,
        fused=True,
        data_format=self.config.data_format,
        is_training=self.is_training,
        scale=True)
    return bn

  def _possible_downsample(self, x, in_filter, out_filter, stride):
    """Downsample the feature map using average pooling, if the filter size
    does not match."""
    if stride[2] > 1:
      with tf.variable_scope("downsample"):
        x = tf.nn.avg_pool(
            x, stride, stride, "VALID", data_format=self.config.data_format)

    if in_filter < out_filter:
      with tf.variable_scope("pad"):
        if self.config.data_format == "NHWC":
          x = tf.pad(x, [[0, 0], [0, 0], [0, 0],
                         [(out_filter - in_filter) // 2,
                          (out_filter - in_filter) // 2]])
        else:
          x = tf.pad(x, [[0, 0], [(out_filter - in_filter) // 2,
                                  (out_filter - in_filter) // 2], [0, 0],
                         [0, 0]])
    return x

  def _residual_inner(self,
                      x,
                      in_filter,
                      out_filter,
                      stride,
                      no_activation=False):
    """Transformation applied on residual units."""
    if self.config.version == "v2":
      with tf.variable_scope("sub1"):
        if not no_activation:
          x = self._batch_norm("bn1", x)
          x = self._relu("relu1", x)
        x = self._conv("conv1", x, 3, in_filter, out_filter, stride)
      with tf.variable_scope("sub2"):
        x = self._batch_norm("bn2", x)
        x = self._relu("relu2", x)
        x = self._conv("conv2", x, 3, out_filter, out_filter, [1, 1, 1, 1])
    else:
      with tf.variable_scope("sub1"):
        x = self._conv("conv1", x, 3, in_filter, out_filter, stride)
        x = self._batch_norm("bn1", x)
        x = self._relu("relu1", x)
      with tf.variable_scope("sub2"):
        x = self._conv("conv2", x, 3, out_filter, out_filter, [1, 1, 1, 1])
        x = self._batch_norm("bn2", x)
    return x

  def _residual(self, x, in_filter, out_filter, stride, no_activation=False):
    """Residual unit with 2 sub layers."""
    orig_x = x
    x = self._residual_inner(
        x, in_filter, out_filter, stride, no_activation=no_activation)
    shortcut = self._possible_downsample(orig_x, in_filter, out_filter, stride)
    x += shortcut
    if self.config.version == "v1":
      x = self._relu("relu3", x)
    return x

  def _bottleneck_residual_inner(self,
                                 x,
                                 in_filter,
                                 out_filter,
                                 stride,
                                 no_activation=False):
    """Transformation applied on bottleneck residual units."""
    if self.config.version == "v2":
      with tf.variable_scope("sub1"):
        if not no_activation:
          x = self._batch_norm("bn1", x)
          x = self._relu("relu1", x)
        x = self._conv("conv1", x, 1, in_filter, out_filter // 4, stride)
      with tf.variable_scope("sub2"):
        x = self._batch_norm("bn2", x)
        x = self._relu("relu2", x)
        x = self._conv("conv2", x, 3, out_filter // 4, out_filter // 4,
                       self._stride_arr(1))
      with tf.variable_scope("sub3"):
        x = self._batch_norm("bn3", x)
        x = self._relu("relu3", x)
        x = self._conv("conv3", x, 1, out_filter // 4, out_filter,
                       self._stride_arr(1))
    else:
      with tf.variable_scope("sub1"):
        x = self._conv("conv1", x, 1, in_filter, out_filter // 4, stride)
        x = self._batch_norm("bn1", x)
        x = self._relu("relu1", x)
      with tf.variable_scope("sub2"):
        x = self._conv("conv2", x, 3, out_filter // 4, out_filter // 4,
                       self._stride_arr(1))
        x = self._batch_norm("bn2", x)
        x = self._relu("relu2", x)
      with tf.variable_scope("sub3"):
        x = self._conv("conv3", x, 1, out_filter // 4, out_filter,
                       self._stride_arr(1))
        x = self._batch_norm("bn3", x)
    return x

  def _possible_bottleneck_downsample(self, x, in_filter, out_filter, stride):
    """Downsample projection layer, if the filter size does not match."""
    if stride[2] > 1 or in_filter != out_filter:
      x = self._conv("project", x, 1, in_filter, out_filter, stride)
    return x

  def _bottleneck_residual(self,
                           x,
                           in_filter,
                           out_filter,
                           stride,
                           no_activation=False):
    """Bottleneck resisual unit with 3 sub layers."""
    orig_x = x
    x = self._bottleneck_residual_inner(
        x, in_filter, out_filter, stride, no_activation=no_activation)
    shortcut = self._possible_bottleneck_downsample(orig_x, in_filter,
                                                    out_filter, stride)
    x += shortcut
    if self.config.version == "v1":
      x = self._relu("relu3", x)
    return x

  def _decay(self):
    """L2 weight decay loss."""
    wd_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    log.info("Weight decay variables")
    [log.info(x) for x in wd_losses]
    log.info("Total length: {}".format(len(wd_losses)))
    if len(wd_losses) > 0:
      return tf.add_n(wd_losses)
    else:
      log.warning("No weight decay variables!")
      return 0.0

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      init_method = "truncated_normal"
      init_param = {"mean": 0, "stddev": np.sqrt(2.0 / n)}
      kernel = self._weight_variable(
          [filter_size, filter_size, in_filters, out_filters],
          init_method=init_method,
          init_param=init_param,
          wd=self.config.weight_decay,
          dtype=self.dtype,
          name="w")
      return tf.nn.conv2d(
          x,
          kernel,
          strides,
          padding="SAME",
          data_format=self.config.data_format)

  def _relu(self, name, x):
    return tf.nn.relu(x, name=name)

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    x_shape = x.get_shape()
    d = x_shape[1]
    w = self._weight_variable(
        [d, out_dim],
        init_method="uniform_scaling",
        init_param={"factor": 1.0},
        wd=self.config.weight_decay,
        dtype=self.dtype,
        name="w")
    b = self._weight_variable(
        [out_dim],
        init_method="constant",
        init_param={"val": 0.0},
        name="b",
        dtype=self.dtype)
    return tf.nn.xw_plus_b(x, w, b)

  def _global_avg_pool(self, x):
    # assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])

  def infer_step(self, sess, inp=None):
    """Run inference."""
    if inp is None:
      feed_data = None
    else:
      feed_data = {self.input: inp}
    return sess.run(self.output, feed_dict=feed_data)

  def eval_step(self, sess, inp=None, label=None):
    if inp is not None and label is not None:
      feed_data = {self.input: inp, self.label: label}
    elif inp is not None:
      feed_data = {self.input: inp}
    elif label is not None:
      feed_data = {self.label: label}
    else:
      feed_data = None
    return sess.run(self.correct)

  def train_step(self, sess, inp=None, label=None):
    """Run training."""
    if inp is not None and label is not None:
      feed_data = {self.input: inp, self.label: label}
    elif inp is not None:
      feed_data = {self.input: inp}
    elif label is not None:
      feed_data = {self.label: label}
    else:
      feed_data = None
    results = sess.run(
        [self.cross_ent, self.train_op, self.bn_update_ops],
        feed_dict=feed_data)
    return results[0]

  @property
  def cost(self):
    return self._cost

  @property
  def train_op(self):
    return self._train_op

  @property
  def bn_update_ops(self):
    if self._bn_update_ops is None:
      bn_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      assert len(bn_ops) > 0
      log.info("BN update ops:")
      [log.info(op) for op in bn_ops]
      self._bn_update_ops = tf.group(*bn_ops)
      log.info("Total number of BN updates: {}".format(len(bn_ops)))
    return self._bn_update_ops

  @property
  def config(self):
    return self._config

  @property
  def lr(self):
    return self._lr

  @property
  def dtype(self):
    return self._dtype

  @property
  def input(self):
    return self._input

  @property
  def output(self):
    return self._output

  @property
  def correct(self):
    return self._correct

  @property
  def label(self):
    return self._label

  @property
  def cross_ent(self):
    return self._cross_ent

  @property
  def global_step(self):
    return self._global_step

  @property
  def grads_and_vars(self):
    return self._grads_and_vars
