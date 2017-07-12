from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import json
import tensorflow as tf

from resnet.utils import logger
from collections import namedtuple
from resnet.models.multi_tower_model import MultiTowerModel
from resnet.models.multi_pass_model import MultiPassModel

log = logger.get()

MODEL_REGISTRY = {}


def RegisterModel(model_name):
  """Registers a configuration."""

  def decorator(f):
    MODEL_REGISTRY[model_name] = f
    return f

  return decorator


def get_model(model_name,
              config,
              is_training=True,
              inference_only=False,
              num_pass=1,
              num_node=1,
              inp=None,
              label=None,
              batch_size=None):
  """Gets a model instance from predefined library.
  Args:
    model_name: String. Name of the model.
    config: Configuration object.
    is_training: Bool. Whether the model is in training mode.
    inference_only: Bool. Whether to only build the inference graph.
    num_pass: Int. Number of forward-backward passes to aggregate.
    num_node: Int. Number of cluster nodes.
    inp: Input tensor, optional, by default a built-in placeholder.
    label: Lable tensor, optional, by default a built-in placeholder.
    batch_size: Int. Specify the batch size. Optional.
  Returns:
    model: A Model instance.
  """
  config_dict = dict(config.__dict__)
  config_copy = json.loads(
      json.dumps(config_dict),
      object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
  key = model_name
  if batch_size is not None:
    batch_size = batch_size // num_pass // num_node
    log.info("Batch size is set to {}".format(batch_size), verbose=0)

  if key not in MODEL_REGISTRY:
    raise ValueError("Unknown model \"{}\"".format(key))

  def _get_model(*args, **kwargs):
    return MODEL_REGISTRY[key](*args, **kwargs)

  if num_pass > 1:
    return MultiPassModelV2(
        config_copy,
        _get_model,
        is_training=is_training,
        num_passes=num_pass,
        batch_size=batch_size,
        inp=inp,
        label=label)
  if num_node > 1:
    assert num_pass == 1, "Not supported"
    return MultiNodeModel(
        config_copy,
        _get_model,
        is_training=is_training,
        num_worker=num_node,
        inp=inp,
        label=label)
  return _get_model(
      config_copy,
      is_training=is_training,
      inp=inp,
      label=label,
      inference_only=inference_only,
      batch_size=batch_size,
      apply_grad=True)


def get_multi_gpu_model(model_name,
                        config,
                        is_training=True,
                        num_replica=1,
                        num_pass=1,
                        num_node=1,
                        inp=None,
                        label=None,
                        batch_size=None,
                        multi_session=False,
                        use_nccl=False):
  """Gets a model instance from predefined library for multi GPU.
  Args:
    model_name: Name of the model.
    config: Configuration object.
    is_training: Bool. Whether the model is in training mode.
    num_replica: Number of parallel training instance (GPU).
    num_pass: Number of serial training steps.
    inference_only: Bool. Whether to only build the inference graph.
    num_pass: Int. Number of forward-backward passes to aggregate.
    num_node: Int. Number of cluster nodes.
    inp: Input tensor, optional, by default a built-in placeholder.
    label: Lable tensor, optional, by default a built-in placeholder.
    batch_size: Int. Specify the batch size. Optional.
    multi_session: Bool. Whether to do the split-graph trick on Reversible Nets.
    use_nccl: Bool. Whether to perform gradient averaging on GPU.
  Returns:
    model: A Model instance.
  """
  config_dict = dict(config.__dict__)
  config_copy = json.loads(
      json.dumps(config_dict),
      object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
  key = model_name
  if key in MODEL_REGISTRY:
    model_cls = MODEL_REGISTRY[key]
  else:
    raise ValueError("Unknown model \"{}\"".format(key))
  if batch_size is not None:
    batch_size = batch_size // num_pass // num_node
    log.info("Batch size is set to {}".format(batch_size), verbose=0)
    log.info("Number of in-graph replica {}".format(num_replica), verbose=0)

  def _get_multi_tower_model(config, **kwargs):
    if use_nccl:
      return MultiTowerModelNCCL(
          config, model_cls, num_replica=num_replica, **kwargs)
    else:
      return MultiTowerModel(
          config, model_cls, num_replica=num_replica, **kwargs)

  if num_replica > 1:
    if num_pass > 1 or num_node > 1:
      apply_grad = False
    else:
      apply_grad = True
    if multi_session:
      assert not use_nccl, "Not supported."
      return MultiTowerMultiSessModel(
          config,
          model_cls,
          num_replica=num_replica,
          is_training=is_training,
          inp=inp,
          label=label,
          batch_size=batch_size,
          apply_grad=apply_grad)
  else:
    raise ValueError("Unacceptable number of replica {}".format(num_replica))

  if num_pass > 1:
    assert not multi_session, "Not supported"
    return MultiPassModelV2(
        config,
        _get_multi_tower_model,
        is_training=is_training,
        num_passes=num_pass,
        inp=inp,
        label=label)
  if num_node > 1:
    assert num_pass == 1, "Not supported"
    return MultiNodeModel(
        config,
        _get_multi_tower_model,
        is_training=is_training,
        num_worker=num_node,
        inp=inp,
        label=label)
  return _get_multi_tower_model(
      config,
      is_training=is_training,
      inp=inp,
      label=label,
      batch_size=batch_size)
