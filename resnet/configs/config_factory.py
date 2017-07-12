from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import json
from collections import namedtuple

CONFIG_REGISTRY = {}


def RegisterConfig(model_name):
  """Registers a configuration."""

  def decorator(f):
    CONFIG_REGISTRY[model_name] = f
    return f

  return decorator


def get_config(model_name):
  """Gets a configuration from predefined library.
  Args:
    model_name: Name of the model.
  """
  key = model_name
  if key in CONFIG_REGISTRY:
    return CONFIG_REGISTRY[key]()
  else:
    raise ValueError("Unknown model \"{}\"".format(key))


def get_config_from_json(path):
  """Gets a configuration from a JSON file."""
  x = json.load(
      open(path, "r"),
      object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
  print(x)
  return x
