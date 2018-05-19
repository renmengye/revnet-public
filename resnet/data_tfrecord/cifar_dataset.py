from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from resnet.data_tfrecord.dataset import Dataset
from resnet.data_tfrecord.data_factory import RegisterDataset


class CifarDataset(Dataset):
  """CIFAR data set."""

  def num_examples_per_epoch(self):
    """Returns the number of examples in the data set."""
    if self.subset == 'train':
      return 45000
    if self.subset == 'validation':
      return 5000
    if self.subset == 'test':
      return 10000
    if self.subset == 'trainval':
      return 50000

  def download_message(self):
    pass

  def available_subsets(self):
    """Returns the list of available subsets."""
    return ['train', 'validation', 'test', 'trainval']


@RegisterDataset("cifar-10")
class Cifar10Dataset(CifarDataset):
  """CIFAR-10 data set."""

  def __init__(self, subset, data_dir):
    super(Cifar10Dataset, self).__init__('CIFAR-10', subset, data_dir)

  def num_classes(self):
    """Returns the number of classes in the data set."""
    return 10


@RegisterDataset("cifar-100")
class Cifar100Dataset(CifarDataset):
  """CIFAR-100 data set."""

  def __init__(self, subset, data_dir):
    super(Cifar100Dataset, self).__init__('CIFAR-100', subset, data_dir)

  def num_classes(self):
    """Returns the number of classes in the data set."""
    return 100
