from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from resnet.utils.factory import Factory


def RegisterDataset(dataset_name):
  return _factory.register(dataset_name)


def get_dataset(name, *args, **kwargs):
  """Gets a dataset instance from predefined library.
  Args:
    name: String. Name of the dataset.
  Returns:
    dataset: A Dataset instance.
  """
  if _factory.has(name):
    return _factory.create(key, *args, **kwargs)
    #return CONFIG_REGISTRY[key]()
  else:
    raise ValueError("Unknown dataset \"{}\"".format(name))
