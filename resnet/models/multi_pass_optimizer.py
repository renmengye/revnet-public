"""
Run multiple forward-backward and then update the average gradients once.
Designed for using smaller number of GPUs to run large experiment with large
mini-batch size (e.g. ImageNet), by sacrificing the running time.

Usage:
>> opt = tf.train.MomentumOptimizer(0.1, momentum=0.9)
>> # This is a wrapper of the original optimizer.
>> opt = MultiPassOptimizer(opt, num_passes=2)
>> grad = opt.compute_gradients()
>> train_op_1 = opt.apply_gradients(grad)  # Gradients being cached.
>> train_op_2 = opt.apply_gradients(grad)  # Weights update happen.
>> ...
>>
>> sess. tf.Session()
>> sess.run(train_op_1, feed_dict={input: input_1})  # Weights not changing.
>> sess.run(train_op_2, feed_dict={input: input_2})  # Weights change now.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from resnet.utils import logger

log = logger.get()


class MultiPassOptimizer(tf.train.Optimizer):

  def __init__(self, opt, num_passes, aggregate_method="cumsum", debug=False):
    """
    Args:
      opt: Inner optimizer to use.
      num_passes: Number of passes before apply the average gradients.
      aggregate_method: Whether to use online sum (cumsum) or store the
      gradients average it altogether (storage), there maybe numerical
      differences.
    """
    self._opt = opt
    self._num_passes = num_passes
    self._count = 0
    self._grad_cache = {}
    self._grad_acc_ops = []
    self._train_op = None
    self._method = aggregate_method  # or "storage".
    self._debug = debug
    if self._method != "cumsum" and self._method != "storage":
      raise Exception("Unknown aggregation method \"{}\"".format(self._method))

  @property
  def opt(self):
    return self._opt

  @property
  def num_passes(self):
    return self._num_passes

  @property
  def train_op(self):
    return self._train_op

  @property
  def grad_cache(self):
    return self._grad_cache

  def compute_gradients(self,
                        loss,
                        var_list=None,
                        gate_gradients=1,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None):
    """Computes the gradients to the variables.
    Register gradient cache variables.
    """
    return self.opt.compute_gradients(
        loss,
        var_list=None,
        gate_gradients=1,
        aggregation_method=None,
        colocate_gradients_with_ops=False,
        grad_loss=None)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """Accumulates gradients."""
    grad_add_ops = []
    if self._count <= self.num_passes - 1:
      for grad, var in grads_and_vars:

        assert self._count == 0 or var in self._grad_cache, "Variable not found."

        # Add to cache.
        if var not in self._grad_cache:
          if self._method == "cumsum":
            self._grad_cache[var] = tf.get_variable(
                var.name.split(":")[0] + "/grad_cache",
                var.get_shape(),
                initializer=tf.constant_initializer(
                    value=0.0, dtype=var.dtype),
                dtype=var.dtype,
                trainable=False)
          else:
            var_shape = [int(ss) for ss in list(var.get_shape())]
            augment_shape = [self.num_passes] + var_shape
            self._grad_cache[var] = tf.get_variable(
                var.name.split(":")[0] + "/grad_cache",
                augment_shape,
                dtype=var.dtype,
                initializer=tf.constant_initializer(
                    value=0.0, dtype=var.dtype),
                trainable=False)

        if grad is not None:
          _grad_cache = self.grad_cache[var]

          if self._method == "cumsum":
            _div = tf.div(grad, self.num_passes)
            _add_op = tf.assign_add(_grad_cache, _div)
            grad_add_ops.append(_add_op)
          else:
            _add = tf.expand_dims(grad, 0)
            _assign_op = tf.scatter_update(_grad_cache, [self._count], _add)
            grad_add_ops.append(_assign_op)
        else:
          if v not in self._grad_cache:
            self._grad_cache[var] = None
    else:
      raise Exception("You cannot call more apply_graidents")
    log.info("Number of grad add ops {}".format(len(grad_add_ops)))
    grad_add_op = tf.group(*grad_add_ops)
    if self._count < self.num_passes - 1:
      final_op = grad_add_op
    else:
      zero_out_ops = []
      with tf.control_dependencies([grad_add_op]):
        if self._method == "cumsum":
          grad_avg = [(tf.identity(gg), var)
                      for var, gg in self._grad_cache.items()]
        else:
          grad_avg = [(tf.reduce_mean(gg, [0]), var)
                      for var, gg in self._grad_cache.items()]

        # Update the weight variables.
        with tf.control_dependencies([grad_add_op]):
          weight_update = self.opt.apply_gradients(
              grad_avg, global_step=global_step, name=name)

        # Zero out gradient cache.
        with tf.control_dependencies([weight_update]):
          for grad, var in grad_avg:
            _grad_cache = self._grad_cache[var]
            if _grad_cache is not None:
              _grad_shape = _grad_cache.get_shape()
              _zeros = tf.zeros(_grad_shape, dtype=_grad_cache.dtype)
              _zero_out_grad = _grad_cache.assign(_zeros)
              zero_out_ops.append(_zero_out_grad)
      if self._debug:  # A hack, remove debug code later.
        final_op = weight_update
      else:
        zero_out_op = tf.group(*zero_out_ops)
        final_op = zero_out_op
    self._count += 1
    return final_op
