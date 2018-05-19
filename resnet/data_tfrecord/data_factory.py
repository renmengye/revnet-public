from __future__ import absolute_import, division, print_function, unicode_literals

from resnet.utils.factory import Factory

_data_factory = Factory()
_inp_factory = Factory()


def RegisterDataset(dataset_name):
  """Registers a configuration."""
  return _data_factory.register(dataset_name)


def RegisterInputPipeline(input_pipeline_name):
  """Registers an input pipeline."""
  return _inp_factory.register(input_pipeline_name)


def get_dataset_cls(dataset_name):
  return _data_factory.get(dataset_name)


def get_input_pipeline_cls(input_pipeline_name):
  return _inp_factory.get(input_pipeline_name)


def get_data_inputs(dataset_name, input_pipeline_name, data_dir, subset,
                    is_training, **kwargs):
  """Gets a data input instance.
  Args:
    dataset_name: String. Name of the dataset.
    subset: String. Subset of the dataset, train or validation.
    is_training: Bool. Whetheor in training mode.
    input_pipeline_name: String. Name of the input pipeline.
  """
  dataset = _data_factory.create(dataset_name, data_dir=data_dir, subset=subset)
  inp = _inp_factory.create(
      input_pipeline_name, dataset, is_training=is_training, **kwargs)
  return inp
