from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import json
import numpy as np
import tensorflow as tf

from collections import namedtuple
from resnet.models.nnlib import concat as _concat
from resnet.models.model_factory import RegisterModel
from resnet.models.resnet_model import ResnetModel
from resnet.utils import logger

log = logger.get()


@RegisterModel("revnet")
class RevnetModel(ResnetModel):

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
    if config.manual_gradients:
      self._wd_hidden = config.wd
      assert self._wd_hidden > 0.0, "Not applying weight decay."
      dd = config.__dict__
      config2 = json.loads(
          json.dumps(config.__dict__),
          object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
      dd = config2.__dict__
      dd["wd"] = 0.0  # Regular weight decay is not used!
      config2 = json.loads(
          json.dumps(dd),
          object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
      assert config2.wd == 0.0, "Weight decay not cleared."
      assert config.wd > 0.0, "Weight decay not cleared."
    else:
      config2 = config
    super(RevNetModel, self).__init__(
        config2,
        is_training=is_training,
        inference_only=inference_only,
        inp=inp,
        label=label,
        dtype=dtype,
        batch_size=batch_size,
        apply_grad=apply_grad)

  def _combine(self, concat, *argv):
    if concat:
      y = _concat(list(argv), axis=3)
    else:
      y = tuple(argv)
    return y

  def _split(self, concat, n_filter, x):
    if concat or type(x) != tuple:
      x1 = x[:, :, :, :n_filter // 2]
      x2 = x[:, :, :, n_filter // 2:]
    else:
      x1, x2 = x
    return x1, x2

  def _residual(self,
                x,
                in_filter,
                out_filter,
                stride,
                no_activation=False,
                concat=False,
                add_bn_ops=True):
    """Residual unit with 2 sub layers.
    Args:
      x: [N, H, W, Ci]. Input activation.
      in_filter: Int. Input number of channels.
      out_filter: Int. Output number of channels.
      stride: Int. Size of the strided convolution.
      no_activation: Bool. Whether to run through BN+ReLU first.
    Returns:
      y: [N, H, W, Cout]. Output activation.
    """
    x1, x2 = self._split(concat, in_filter, x)
    with tf.variable_scope("f"):
      f_x2 = self._residual_inner(
          x2,
          in_filter // 2,
          out_filter // 2,
          stride,
          no_activation=no_activation,
          add_bn_ops=add_bn_ops)
    x1_ = self._possible_downsample(x1, in_filter // 2, out_filter // 2, stride)
    x2_ = self._possible_downsample(x2, in_filter // 2, out_filter // 2, stride)
    y1 = f_x2 + x1_
    with tf.variable_scope("g"):
      f_y1 = self._residual_inner(
          y1,
          out_filter // 2,
          out_filter // 2,
          self._stride_arr(1),
          add_bn_ops=add_bn_ops)
    y2 = f_y1 + x2_
    return self._combine(concat, y1, y2)

  def _bottleneck_residual(self,
                           x,
                           in_filter,
                           out_filter,
                           stride,
                           no_activation=False,
                           concat=False,
                           add_bn_ops=True):
    """Bottleneck resisual unit with 3 sub layers.
    Args:
      x: [N, H, W, Ci]. Input activation.
      in_filter: Int. Input number of channels.
      out_filter: Int. Output number of channels.
      stride: Int. Size of the strided convolution.
      no_activation: Bool. Whether to run through BN+ReLU first.
    Returns:
      y: [N, H, W, Cout]. Output activation.
    """
    x1, x2 = self._split(concat, in_filter, x)
    with tf.variable_scope("f"):
      f_x2 = self._bottleneck_residual_inner(
          x2,
          in_filter // 2,
          out_filter // 2,
          stride,
          no_activation=no_activation,
          add_bn_ops=add_bn_ops)
    with tf.variable_scope("x1_bottleneck"):
      x1_ = self._possible_bottleneck_downsample(x1, in_filter // 2,
                                                 out_filter // 2, stride)
    with tf.variable_scope("x2_bottleneck"):
      x2_ = self._possible_bottleneck_downsample(x2, in_filter // 2,
                                                 out_filter // 2, stride)
    y1 = f_x2 + x1_
    with tf.variable_scope("g"):
      f_y1 = self._bottleneck_residual_inner(
          y1,
          out_filter // 2,
          out_filter // 2,
          self._stride_arr(1),
          add_bn_ops=add_bn_ops)
    y2 = f_y1 + x2_
    return self._combine(concat, y1, y2)

  def _residual_backward(self, y, n_filter, concat=False):
    """Reconstruction of input activation with 2 sub layers.
    Args:
      y: [N, H, W, C]. Output activation.
      n_filter: Int. Number of channels.
    Returns:
      x: [N, H, W, C]. Input activation.
    """
    y1, y2 = self._split(concat, n_filter, y)
    with tf.variable_scope("g"):
      f_y1 = self._residual_inner(
          y1,
          n_filter // 2,
          n_filter // 2,
          self._stride_arr(1),
          add_bn_ops=False)
    x2 = y2 - f_y1
    with tf.variable_scope("f"):
      f_x2 = self._residual_inner(
          x2,
          n_filter // 2,
          n_filter // 2,
          self._stride_arr(1),
          add_bn_ops=False)
    x1 = y1 - f_x2
    return self._combine(concat, x1, x2)

  def _bottleneck_residual_backward(self, y, n_filter, concat=False):
    """Reconstruction of input activation with 3 sub layers.
    Args:
      y: [N, H, W, C]. Output activation.
      n_filter: Int. Number of channels.
    Returns:
      x: [N, H, W, C]. Input activation.
    """
    y1, y2 = self._split(concat, n_filter, y)
    with tf.variable_scope("g"):
      f_y1 = self._bottleneck_residual_inner(
          y1,
          n_filter // 2,
          n_filter // 2,
          self._stride_arr(1),
          add_bn_ops=False)
    x2 = y2 - f_y1
    with tf.variable_scope("f"):
      f_x2 = self._bottleneck_residual_inner(
          x2,
          n_filter // 2,
          n_filter // 2,
          self._stride_arr(1),
          add_bn_ops=False)
    x1 = y1 - f_x2
    return self._combine(concat, x1, x2)

  def _residual_grad(self,
                     x,
                     dy,
                     in_filter,
                     out_filter,
                     stride,
                     no_activation=False,
                     concat=False):
    """Gradients without referring to the stored activation.
    Args:
      x: [N, H, W, Cin]. Input activation.
      dy: [N, H, W, Cout]. Output gradient.
      in_filter: Int. Input number of channels.
      out_filter: Int. Output number of channels.
      stride: Int. Size of the strided convolution.
      no_activation: Bool. Whether to run through BN+ReLU first.
    Returns:
      dx: [N, H, W, Cin]. Input gradient.
      w: List of variables.
      dw: List of gradients towards the variables.
    """
    x1, x2 = self._split(concat, in_filter, x)
    x1, x2 = tf.stop_gradient(x1), tf.stop_gradient(x2)
    dy1, dy2 = self._split(concat, out_filter, dy)

    if self.config.use_bottleneck:
      y1_, y2_ = self._bottleneck_residual(
          (x1, x2),
          in_filter,
          out_filter,
          stride=stride,
          no_activation=no_activation,
          concat=False,
          add_bn_ops=False)
      with tf.variable_scope("x2_bottleneck"):
        x2_ = self._possible_bottleneck_downsample(x2, in_filter // 2,
                                                   out_filter // 2, stride)
    else:
      y1_, y2_ = self._residual(
          (x1, x2),
          in_filter,
          out_filter,
          stride=stride,
          no_activation=no_activation,
          concat=False,
          add_bn_ops=False)
      x2_ = self._possible_downsample(x2, in_filter // 2, out_filter // 2,
                                      stride)

    # F function weights.
    if no_activation:
      fw_names = []
    else:
      fw_names = ["f/sub1/bn1/beta", "f/sub1/bn1/gamma"]
    fw_names.append("f/sub1/conv1/w")
    num_layers = 3 if self.config.use_bottleneck else 2
    for ii in range(2, num_layers + 1):
      fw_names.append("f/sub{}/bn{}/beta".format(ii, ii))
      fw_names.append("f/sub{}/bn{}/gamma".format(ii, ii))
      fw_names.append("f/sub{}/conv{}/w".format(ii, ii))
    fw_list = map(lambda x: tf.get_variable(x), fw_names)

    # G function weights.
    gw_names = []
    for ii in range(1, num_layers + 1):
      gw_names.append("g/sub{}/bn{}/beta".format(ii, ii))
      gw_names.append("g/sub{}/bn{}/gamma".format(ii, ii))
      gw_names.append("g/sub{}/conv{}/w".format(ii, ii))
    gw_list = map(lambda x: tf.get_variable(x), gw_names)

    # Downsample function weights.
    b1w_names = []
    b2w_names = []
    if self.config.use_bottleneck:
      b1w_names.append("x1_bottleneck/project/w")
      b2w_names.append("x2_bottleneck/project/w")

      def try_get_variable(x):
        try:
          v = tf.get_variable(x)
        except Exception as e:
          return None
        return v

      b1w_list = filter(lambda x: x is not None, map(try_get_variable,
                                                     b1w_names))
      b2w_list = filter(lambda x: x is not None, map(try_get_variable,
                                                     b2w_names))

    dd1 = tf.gradients(y2_, [y1_] + gw_list, dy2, gate_gradients=True)
    dy2_y1 = dd1[0]
    dy1_plus = dy2_y1 + dy1
    dgw = dd1[1:]
    dd2 = tf.gradients(y1_, [x1, x2] + fw_list, dy1_plus, gate_gradients=True)
    dx1 = dd2[0]
    dx2 = dd2[1]
    dfw = dd2[2:]
    dx2 += tf.gradients(x2_, x2, dy2, gate_gradients=True)[0]

    dw_list = list(dfw) + list(dgw)
    w_list = list(fw_list) + list(gw_list)

    # Downsample function gradients.
    if self.config.use_bottleneck:
      if len(b1w_list) > 0:
        db1w = tf.gradients(y1_, b1w_list, dy1_plus, gate_gradients=True)
        dw_list += list(db1w)
        w_list += list(b1w_list)
      if len(b2w_list) > 0:
        db2w = tf.gradients(y2_, b2w_list, dy2, gate_gradients=True)
        dw_list += list(db2w)
        w_list += list(b2w_list)

    # Inject dw dependency.
    with tf.control_dependencies(dw_list):
      dx = self._combine(concat, tf.identity(dx1), tf.identity(dx2))
    return dx, w_list, dw_list

  def _init_conv_grad(self, y, dy):
    """Gradients for initial convolutions."""
    w_init = tf.get_variable("init/init_conv/w")
    beta_init = tf.get_variable("init/init_bn/beta")
    gamma_init = tf.get_variable("init/init_bn/gamma")
    w_list = [beta_init, gamma_init, w_init]
    dw_list = tf.gradients(y, w_list, dy)
    return w_list, dw_list

  def _compute_gradients(self, cost):
    """Computes gradients.
    Args:
      cost: Loss function.
    Returns:
      grads_and_vars: List of tuple of gradients and variables.
    """
    config = self.config
    if not config.manual_gradients:
      return super(RevNetModel, self)._compute_gradients(cost)
    log.warning("Manually building gradient graph.")
    g = tf.get_default_graph()
    tf.get_variable_scope().reuse_variables()
    num_stages = len(self.config.num_residual_units)
    beta_final = tf.get_variable("unit_last/final_bn/beta")
    gamma_final = tf.get_variable("unit_last/final_bn/gamma")
    w_final = tf.get_variable("logit/w")
    b_final = tf.get_variable("logit/b")
    filters = [ff for ff in self.config.filters]  # Copy filter config.

    if config.use_bottleneck:
      res_func = self._bottleneck_residual_backward
      # For CIFAR-10 it's [16, 16, 32, 64] => [16, 64, 128, 256]
      for ii in range(1, len(filters)):
        filters[ii] *= 4
    else:
      res_func = self._residual_backward

    grads_list = []
    vars_list = []
    var_final = [beta_final, gamma_final, w_final, b_final]

    h1, h2 = self._saved_hidden[-1]
    h1, h2 = tf.stop_gradient(h1), tf.stop_gradient(h2)
    h = _concat([h1, h2], axis=3)
    with tf.variable_scope("unit_last"):
      h = self._batch_norm("final_bn", h, add_ops=False)
      h = self._relu("final_relu", h)
    h = self._global_avg_pool(h)
    with tf.variable_scope("logit"):
      logits = self._fully_connected(h, config.num_classes)
    with tf.variable_scope("costs"):
      xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=self.label)
      cost = tf.reduce_mean(xent, name="xent")

    _grads = tf.gradients(cost, [h1, h2] + var_final, gate_gradients=True)
    dh1, dh2 = _grads[0], _grads[1]
    _grads = _grads[2:]
    # Injected dependency.
    with tf.control_dependencies(_grads):
      h_grad = (tf.identity(dh1), tf.identity(dh2))
    grads_list.extend(_grads)
    # grads_list.extend(_grads[2:])
    vars_list.extend(var_final)

    h1, h2 = self._saved_hidden[-1]
    h1, h2 = tf.stop_gradient(h1), tf.stop_gradient(h2)
    h = (h1, h2)

    # New version, using single for-loop.
    ss = num_stages - 1
    ii = config.num_residual_units[ss] - 1
    nlayers = sum(config.num_residual_units)
    for ll in range(nlayers - 1, -1, -1):
      no_activation = False
      if ii == 0:
        in_filter = filters[ss]
        stride = self._stride_arr(self.config.strides[ss])
        if ss == 0:
          no_activation = True
      else:
        in_filter = filters[ss + 1]
        stride = self._stride_arr(1)
      out_filter = filters[ss + 1]

      with tf.variable_scope("unit_{}_{}".format(ss + 1, ii)):

        # Reconstruct input.
        if ii == 0:
          h = self._saved_hidden[ss]
        else:
          h = res_func(h, out_filter)

        # Rerun the layer, and get gradients.
        h_grad, w_list, w_grad = self._residual_grad(
            h,
            h_grad,
            in_filter,
            out_filter,
            stride,
            no_activation=no_activation)

        grads_list.extend(w_grad)
        vars_list.extend(w_list)

      # Counter.
      if ii == 0:
        ss -= 1
        ii = config.num_residual_units[ss] - 1
      else:
        ii -= 1

    h_grad = _concat(h_grad, axis=3)
    var_init, _grads = self._init_conv_grad(h, h_grad)
    grads_list.extend(_grads)
    vars_list.extend(var_init)

    # Add weight decay.
    def add_wd(x):
      g, w = x[0], x[1]
      assert self._wd_hidden > 0.0, "Not applying weight decay"
      if w.name.endswith("w:0") and self._wd_hidden > 0.0:
        log.info("Adding weight decay {:.4e} for variable {}".format(
            self._wd_hidden, x[1].name))
        return g + self._wd_hidden * w, w
      else:
        return g, w

    # Always gate gradients to avoid unwanted behaviour.
    return map(add_wd, zip(tf.tuple(grads_list), vars_list))
