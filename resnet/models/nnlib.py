from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from resnet.utils import logger

log = logger.get()


def weight_variable(shape,
                    init_method=None,
                    dtype=tf.float32,
                    init_param=None,
                    wd=None,
                    name=None,
                    trainable=True,
                    seed=0):
  """Declares a variable.

    Args:
        shape: Shape of the weights, list of int.
        init_method: Initialization method, "constant" or "truncated_normal".
        init_param: Initialization parameters, dictionary.
        wd: Weight decay, float.
        name: Name of the variable, str.
        trainable: Whether the variable can be trained, bool.

    Returns:
        var: Declared variable.
    """
  if dtype != tf.float32:
    log.warning("Not using float32, currently using {}".format(dtype))
  if init_method is None:
    initializer = tf.zeros_initializer(shape, dtype=dtype)
  elif init_method == "truncated_normal":
    if "mean" not in init_param:
      mean = 0.0
    else:
      mean = init_param["mean"]
    if "stddev" not in init_param:
      stddev = 0.1
    else:
      stddev = init_param["stddev"]
    log.info("Normal initialization std {:.3e}".format(stddev))
    initializer = tf.truncated_normal_initializer(
        mean=mean, stddev=stddev, seed=seed, dtype=dtype)
  elif init_method == "uniform_scaling":
    if "factor" not in init_param:
      factor = 1.0
    else:
      factor = init_param["factor"]
    log.info("Uniform initialization scale {:.3e}".format(factor))
    initializer = tf.uniform_unit_scaling_initializer(
        factor=factor, seed=seed, dtype=dtype)
  elif init_method == "constant":
    if "val" not in init_param:
      value = 0.0
    else:
      value = init_param["val"]
    initializer = tf.constant_initializer(value=value, dtype=dtype)
  elif init_method == "xavier":
    initializer = tf.contrib.layers.xavier_initializer(
        uniform=False, seed=seed, dtype=dtype)
  else:
    raise ValueError("Non supported initialization method!")
  try:
    shape_int = [int(ss) for ss in shape]
    log.info("Weight shape {}".format(shape_int))
  except:
    pass
  if wd is not None:
    if wd > 0.0:
      reg = lambda x: tf.multiply(tf.nn.l2_loss(x), wd)
      log.info("Weight decay {}".format(wd))
    else:
      log.warning("No weight decay")
      reg = None
  else:
    log.warning("No weight decay")
    reg = None
  var = tf.get_variable(
      name,
      shape,
      initializer=initializer,
      regularizer=reg,
      dtype=dtype,
      trainable=trainable)
  log.info("Initialized weight {}".format(var.name))
  return var


def weight_variable_cpu(shape,
                        init_method=None,
                        dtype=tf.float32,
                        init_param=None,
                        wd=None,
                        name=None,
                        trainable=True,
                        seed=0):
  """Declares variables on CPU."""
  with tf.device("/cpu:0"):
    return weight_variable(
        shape,
        init_method=init_method,
        dtype=dtype,
        init_param=init_param,
        wd=wd,
        name=name,
        trainable=trainable,
        seed=seed)


def concat(x, axis):
  if tf.__version__.startswith("0"):
    return tf.concat(axis, x)
  else:
    return tf.concat(x, axis=axis)


def split(x, num, axis):
  if tf.__version__.startswith("0"):
    return tf.split(axis, num, x)
  else:
    return tf.split(x, num, axis)


def stack(x):
  if tf.__version__.startswith("0"):
    return tf.pack(x)
  else:
    return tf.stack(x)


def cnn(x,
        filter_size,
        strides,
        pool_fn,
        pool_size,
        pool_strides,
        act_fn,
        dtype=tf.float32,
        add_bias=True,
        wd=None,
        init_std=None,
        init_method=None,
        scope="cnn",
        trainable=True):
  """Builds a convolutional neural networks.
    Each layer contains the following operations:
        1) Convolution, y = w * x.
        2) Additive bias (optional), y = w * x + b.
        3) Activation function (optional), y = g( w * x + b ).
        4) Pooling (optional).

    Args:
        x: Input variable.
        filter_size: Shape of the convolutional filters, list of 4-d int.
        strides: Convolution strides, list of 4-d int.
        pool_fn: Pooling functions, list of N callable objects.
        pool_size: Pooling field size, list of 4-d int.
        pool_strides: Pooling strides, list of 4-d int.
        act_fn: Activation functions, list of N callable objects.
        add_bias: Whether adding bias or not, bool.
        wd: Weight decay, float.
        scope: Scope of the model, str.
    """
  num_layer = len(filter_size)
  h = x
  with tf.variable_scope(scope):
    for ii in range(num_layer):
      with tf.variable_scope("layer_{}".format(ii)):
        if init_method is not None and init_method[ii]:
          w = weight_variable_cpu(
              filter_size[ii],
              init_method=init_method[ii],
              dtype=dtype,
              init_param={"mean": 0.0,
                          "stddev": init_std[ii]},
              wd=wd,
              name="w",
              trainable=trainable)
        else:
          w = weight_variable_cpu(
              filter_size[ii],
              init_method="truncated_normal",
              dtype=dtype,
              init_param={"mean": 0.0,
                          "stddev": init_std[ii]},
              wd=wd,
              name="w",
              trainable=trainable)

        if add_bias:
          b = weight_variable_cpu(
              [filter_size[ii][3]],
              init_method="constant",
              dtype=dtype,
              init_param={"val": 0},
              # wd=wd,       ####### Change this back if it changes anything!!!
              name="b",
              trainable=trainable)
        h = tf.nn.conv2d(h, w, strides=strides[ii], padding="SAME", name="conv")
        if add_bias:
          h = tf.add(h, b, name="conv_bias")
        if act_fn[ii] is not None:
          h = act_fn[ii](h, name="act")
        if pool_fn[ii] is not None:
          h = pool_fn[ii](h,
                          pool_size[ii],
                          strides=pool_strides[ii],
                          padding="SAME",
                          name="pool")
  return h


def mlp(x,
        dims,
        is_training=True,
        act_fn=None,
        dtype=tf.float32,
        add_bias=True,
        wd=None,
        init_std=None,
        init_method=None,
        scope="mlp",
        dropout=None,
        trainable=True):
  """Builds a multi-layer perceptron.
    Each layer contains the following operations:
        1) Linear transformation, y = w^T x.
        2) Additive bias (optional), y = w^T x + b.
        3) Activation function (optional), y = g( w^T x + b )
        4) Dropout (optional)

    Args:
        x: Input variable.
        dims: Layer dimensions, list of N+1 int.
        act_fn: Activation functions, list of N callable objects.
        add_bias: Whether adding bias or not, bool.
        wd: Weight decay, float.
        scope: Scope of the model, str.
        dropout: Whether to apply dropout, None or list of N bool.
    """
  num_layer = len(dims) - 1
  h = x
  with tf.variable_scope(scope):
    for ii in range(num_layer):
      with tf.variable_scope("layer_{}".format(ii)):
        dim_in = dims[ii]
        dim_out = dims[ii + 1]

        if init_method is not None and init_method[ii]:
          w = weight_variable_cpu(
              [dim_in, dim_out],
              init_method=init_method[ii],
              dtype=dtype,
              init_param={"mean": 0.0,
                          "stddev": init_std[ii]},
              wd=wd,
              name="w",
              trainable=trainable)
        else:
          w = weight_variable_cpu(
              [dim_in, dim_out],
              init_method="truncated_normal",
              dtype=dtype,
              init_param={"mean": 0.0,
                          "stddev": init_std[ii]},
              wd=wd,
              name="w",
              trainable=trainable)

        if add_bias:
          b = weight_variable_cpu(
              [dim_out],
              init_method="constant",
              dtype=dtype,
              init_param={"val": 0.0},
              # wd=wd,       ####### Change this back if it changes anything!!!
              name="b",
              trainable=trainable)

        h = tf.matmul(h, w, name="linear")
        if add_bias:
          h = tf.add(h, b, name="linear_bias")
        if act_fn and act_fn[ii] is not None:
          h = act_fn[ii](h)
        if dropout is not None and dropout[ii]:
          log.info("Apply dropout 0.5")
          if is_training:
            keep_prob = 0.5
          else:
            keep_prob = 1.0
          h = tf.nn.dropout(h, keep_prob=keep_prob)
  return h


def batch_norm(x,
               is_training,
               gamma=None,
               beta=None,
               axes=[0, 1, 2],
               eps=1e-10,
               name="bn_out",
               decay=0.99,
               dtype=tf.float32):
  """Applies batch normalization.
    Collect mean and variances on x except the last dimension. And apply
    normalization as below:
    x_ = gamma * (x - mean) / sqrt(var + eps) + beta

    Args:
      x: Input tensor, [B, ...].
      n_out: Integer, depth of input variable.
      gamma: Scaling parameter.
      beta: Bias parameter.
      axes: Axes to collect statistics.
      eps: Denominator bias.

    Returns:
      normed: Batch-normalized variable.
      mean: Mean used for normalization (optional).
  """
  n_out = x.get_shape()[-1]
  try:
    n_out = int(n_out)
    shape = [n_out]
  except:
    shape = None
  emean = tf.get_variable(
      "ema_mean",
      shape=shape,
      trainable=False,
      dtype=dtype,
      initializer=tf.constant_initializer(
          0.0, dtype=dtype))
  evar = tf.get_variable(
      "ema_var",
      shape=shape,
      trainable=False,
      dtype=dtype,
      initializer=tf.constant_initializer(
          1.0, dtype=dtype))
  if is_training:
    mean, var = tf.nn.moments(x, axes, name="moments")
    ema_mean_op = tf.assign_sub(emean, (emean - mean) * (1 - decay))
    ema_var_op = tf.assign_sub(evar, (evar - var) * (1 - decay))
    normed = tf.nn.batch_normalization(
        x, mean, var, beta, gamma, eps, name=name)
    return normed, [ema_mean_op, ema_var_op]
  else:
    normed = tf.nn.batch_normalization(
        x, emean, evar, beta, gamma, eps, name=name)
    return normed, None


def layer_norm(x,
               gamma=None,
               beta=None,
               axes=[1, 2, 3],
               eps=1e-3,
               scope="ln",
               name="ln_out",
               return_mean=False):
  """Applies layer normalization.
    Collect mean and variances on x except the first dimension. And apply
    normalization as below:
        x_ = gamma * (x - mean) / sqrt(var + eps)

    Args:
        x: Input tensor, [B, ...].
        axes: Axes to collect statistics.
        gamma: Scaling parameter.
        beta: Bias parameter.
        eps: Denominator bias.
        return_mean: Whether to also return the computed mean.

    Returns:
        normed: Layer-normalized variable.
        mean: Mean used for normalization (optional).
    """
  with tf.variable_scope(scope):
    x_shape = [x.get_shape()[-1]]
    mean, var = tf.nn.moments(x, axes, name='moments', keep_dims=True)
    normed = (x - mean) / tf.sqrt(eps + var)
    if gamma is not None:
      normed *= gamma
    if beta is not None:
      normed += beta
    normed = tf.identity(normed, name=name)
  if return_mean:
    return normed, mean
  else:
    return normed


def div_norm_2d(x,
                sum_window,
                sup_window,
                gamma=None,
                beta=None,
                eps=1.0,
                scope="dn",
                name="dn_out",
                return_mean=False):
  """Applies divisive normalization on CNN feature maps.
    Collect mean and variances on x on a local window across channels. And apply
    normalization as below:
        x_ = gamma * (x - mean) / sqrt(var + eps)

    Args:
        x: Input tensor, [B, H, W, C].
        sum_window: Summation window size, [H_sum, W_sum].
        sup_window: Suppression window size, [H_sup, W_sup].
        gamma: Scaling parameter.
        beta: Bias parameter.
        eps: Denominator bias.
        return_mean: Whether to also return the computed mean.

    Returns:
        normed: Divisive-normalized variable.
        mean: Mean used for normalization (optional).
    """
  with tf.variable_scope(scope):
    w_sum = tf.ones(sum_window + [1, 1]) / np.prod(np.array(sum_window))
    w_sup = tf.ones(sup_window + [1, 1]) / np.prod(np.array(sum_window))
    x_mean = tf.reduce_mean(x, [3], keep_dims=True)
    x_mean = tf.nn.conv2d(x_mean, w_sum, strides=[1, 1, 1, 1], padding='SAME')
    normed = x - x_mean
    x2 = tf.square(normed)
    x2_mean = tf.reduce_mean(x2, [3], keep_dims=True)
    x2_mean = tf.nn.conv2d(x2_mean, w_sup, strides=[1, 1, 1, 1], padding='SAME')
    denom = tf.sqrt(x2_mean + eps)
    normed = normed / denom
    if gamma is not None:
      normed *= gamma
    if beta is not None:
      normed += beta
    normed = tf.identity(normed, name=name)
  if return_mean:
    return normed, x_mean
  else:
    return normed
