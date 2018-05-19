"""Multi-tower CNN for training parallel GPU jobs."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import tensorflow as tf

from resnet.models.nnlib import concat, split, stack
from resnet.utils import logger

log = logger.get()


class MultiTowerModel(object):

  def __init__(self,
               config,
               tower_cls,
               is_training=True,
               inference_only=False,
               num_replica=2,
               inp=None,
               label=None,
               apply_grad=True,
               batch_size=None):
    self._config = config
    self._is_training = is_training
    self._inference_only = inference_only
    self._num_replica = num_replica
    self._apply_grad = apply_grad
    self._tower_cls = tower_cls
    self._batch_size = batch_size

    # Input.
    if inp is None:
      x = tf.placeholder(
          self.dtype,
          [batch_size, config.height, config.width, config.num_channel])
    else:
      x = inp
    if label is None:
      y = tf.placeholder(tf.int32, [batch_size])
    else:
      y = label
    self._bn_update_ops = None
    self._input = x
    # Make sure that the labels are in reasonable range.
    # with tf.control_dependencies(
    #     [tf.assert_greater_equal(y, 0), tf.assert_less(y, config.num_classes)]):
    #   self._label = tf.identity(y)
    self._label = y
    self._towers = []
    self._build_towers()

  def _average_gradients(self, tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
      Note that this function provides a synchronization point across all towers.
      Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
          is over individual gradients. The inner list is over the gradient
          calculation for each tower.
      Returns:
         List of pairs of (gradient, variable) where the gradient has been averaged
         across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
      # Note that each grad_and_vars looks like the following:
      #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
      grads = []
      for g, v in grad_and_vars:
        # Add 0 dimension to the gradients to represent the tower.
        if g is None:
          log.warning("No gradient for variable \"{}\"".format(v.name))
          grads.append(None)
          break
        else:
          expanded_g = tf.expand_dims(g, 0)
          grads.append(expanded_g)

      # Average over the "tower" dimension.
      if grads[0] is None:
        grad = None
      else:
        grad = concat(grads, axis=0)
        grad = tf.reduce_mean(grad, 0)

      # Keep in mind that the Variables are redundant because they are shared
      # across towers. So .. we will just return the first tower"s pointer to
      # the Variable.
      v = grad_and_vars[0][1]
      grad_and_var = (grad, v)
      average_grads.append(grad_and_var)
    return average_grads

  @property
  def dtype(self):
    tensor_type = os.getenv("TF_DTYPE", "float32")
    if tensor_type == "float32":
      return tf.float32
    elif tensor_type == "float64":
      return tf.float64
    else:
      raise Exception("Unknown tensor type {}".format(tensor_type))

  def assign_weights(self, weights):
    return self._towers[0].assign_weights(weights)

  def get_weights(self):
    return self._towers[0].get_weights()

  def _get_device(self, device_name="cpu", device_id=0):
    return "/{}:{:d}".format(device_name, device_id)

  def _build_towers(self):
    # Calculate the gradients for each model tower.
    config = self.config
    tower_grads = []
    op_list = []

    with tf.device(self._get_device("cpu", 0)):
      inputs = split(self.input, self.num_replica, axis=0)
      labels = split(self.label, self.num_replica, axis=0)
      outputs = []
      costs = []
      cross_ents = []
      tower_grads_and_vars = []
      for ii in range(self.num_replica):
        device = self._get_device("gpu", ii)
        with tf.device(device):
          with tf.name_scope("%s_%d" % ("replica", ii)) as scope:
            tower_ = self._tower_cls(
                config,
                is_training=self.is_training,
                inference_only=True,
                inp=inputs[ii],
                label=labels[ii],
                batch_size=self._batch_size,
                idx=ii)
            outputs.append(tower_.output)
            cross_ents.append(tower_.cross_ent)
            costs.append(tower_.cost)
            self._towers.append(tower_)

            # Only update BN for the last tower.
            if ii < self.num_replica - 1 and self.is_training:
              bn_ops = tf.get_collection_ref(tf.GraphKeys.UPDATE_OPS)
              del bn_ops[:]

            if self.is_training:
              # Calculate the gradients for the batch of data on this tower.
              wd_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
              if len(wd_losses) > 0:
                log.info("Replica {}, Weight decay variables: {}".format(
                    ii, wd_losses))
                log.info(
                    "Replica {}, Number of weight decay variables: {}".format(
                        ii, len(wd_losses)))
              tower_grads_and_vars.append(
                  tower_._compute_gradients(tower_.cost))

            log.info("Replica {} built".format(ii), verbose=0)

            # Reuse variables for the next tower.
            tf.get_variable_scope().reuse_variables()

      self._output = concat(outputs, axis=0)
      self._output_idx = tf.cast(tf.argmax(self._output, axis=1), tf.int32)
      self._correct = tf.to_float(
          tf.equal(tf.cast(self._output_idx, self.label.dtype), self.label))
      self._cost = tf.reduce_mean(stack(costs))
      self._cross_ent = tf.reduce_mean(stack(cross_ents))
      if not self.is_training or self.inference_only:
        return

      grads_and_vars = self._average_gradients(tower_grads_and_vars)
      self._tower_grads_and_vars = tower_grads_and_vars
      self._grads_and_vars = grads_and_vars

      if self._apply_grad:
        tf.get_variable_scope()._reuse = None
        global_step = tf.get_variable(
            "global_step", [],
            initializer=tf.constant_initializer(0.0),
            trainable=False,
            dtype=self.dtype)
        self._lr = tf.train.piecewise_constant(
            global_step,
            [float(ss) for ss in self.config.learn_rate_decay_steps],
            [self.config.learn_rate] + list(self.config.learn_rate_list))
        self._global_step = global_step
        opt = tf.train.MomentumOptimizer(self.lr, momentum=self.config.momentum)
        train_op = opt.apply_gradients(grads_and_vars, global_step=global_step)
        self._train_op = train_op

  @property
  def bn_update_ops(self):
    if self._bn_update_ops is None:
      bn_ops = tf.get_collection_ref(tf.GraphKeys.UPDATE_OPS)
      log.info("BN update ops:")
      [log.info(op) for op in bn_ops]
      self._bn_update_ops = tf.group(*bn_ops)
      log.info("Total number of BN updates: {}".format(len(bn_ops)))
    return self._bn_update_ops

  @property
  def global_step(self):
    return self._global_step

  def infer_step(self, sess, inp=None):
    """Run inference."""
    if inp is not None:
      feed_data = {self.input: inp}
    else:
      feed_data = None
    return sess.run(self.output, feed_dict={self.input: inp})

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
  def grads_and_vars(self):
    return self._grads_and_vars

  @property
  def tower_grads_and_vars(self):
    return self._tower_grads_and_vars

  @property
  def config(self):
    return self._config

  @property
  def is_training(self):
    return self._is_training

  @property
  def inference_only(self):
    return self._inference_only

  @property
  def num_replica(self):
    return self._num_replica

  @property
  def cost(self):
    return self._cost

  @property
  def cross_ent(self):
    return self._cross_ent

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op
