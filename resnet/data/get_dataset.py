from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from resnet.data.cifar10 import CIFAR10Dataset
from resnet.data.cifar100 import CIFAR100Dataset
from resnet.utils.batch_iter import BatchIterator
from resnet.utils.concurrent_batch_iter import ConcurrentBatchIterator


def get_dataset(name,
                split,
                data_aug=True,
                batch_size=100,
                cycle=True,
                prefetch=True,
                shuffle=True,
                num_batches=-1):
  """Gets a dataset.

  Args:
      name: "cifar-10" or "cifar-100".
      split: "train", "traintrain", "trainval", or "test".

  Returns:
      dp: Dataset Iterator.
  """
  if name == "cifar-10":
    dp = CIFAR10Dataset(
        "data/cifar-10", split, data_aug=data_aug, whiten=False, div255=False)
    return get_iter(
        dp,
        batch_size=batch_size,
        shuffle=shuffle,
        cycle=cycle,
        prefetch=prefetch,
        num_worker=20,
        queue_size=300,
        num_batches=num_batches)
  elif name == "cifar-100":
    dp = CIFAR100Dataset(
        "data/cifar-100", split, data_aug=data_aug, whiten=False, div255=False)
    return get_iter(
        dp,
        batch_size=batch_size,
        shuffle=shuffle,
        cycle=cycle,
        prefetch=prefetch,
        num_worker=20,
        queue_size=300,
        num_batches=num_batches)
  else:
    raise Exception("Unknown dataset {}".format(dataset))


def get_iter(dataset,
             batch_size=100,
             shuffle=False,
             cycle=False,
             log_epoch=-1,
             seed=0,
             prefetch=False,
             num_worker=20,
             queue_size=300,
             num_batches=-1):
  """Gets a data iterator.

  Args:
      dataset: Dataset object.
      batch_size: Mini-batch size.
      shuffle: Whether to shuffle the data.
      cycle: Whether to stop after one full epoch.
      log_epoch: Log progress after how many iterations.

  Returns:
      b: Batch iterator object.
  """
  b = BatchIterator(
      dataset.get_size(),
      batch_size=batch_size,
      shuffle=shuffle,
      cycle=cycle,
      get_fn=dataset.get_batch_idx,
      log_epoch=log_epoch,
      seed=seed,
      num_batches=num_batches)
  if prefetch:
    b = ConcurrentBatchIterator(
        b, max_queue_size=queue_size, num_threads=num_worker, log_queue=-1)
  return b
