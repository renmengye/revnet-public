"""Input pipeline abstract class for TFRecord."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import abstractmethod

import tensorflow as tf

flags = tf.flags
flags.DEFINE_integer('num_preprocess_threads', 4,
                     """Number of preprocessing threads per tower. """
                     """Please make this a multiple of 4.""")
flags.DEFINE_integer('num_readers', 4,
                     """Number of parallel readers during train.""")
# Images are preprocessed asynchronously using multiple threads specified by
# --num_preprocss_threads and the resulting processed images are stored in a
# random shuffling queue. The shuffling queue dequeues --batch_size images
# for processing on a given Inception tower. A larger shuffling queue guarantees
# better mixing across examples within a batch and results in slightly higher
# predictive performance in a trained model. Empirically,
# --input_queue_memory_factor=16 works well. A value of 16 implies a queue size
# of 1024*16 images. Assuming RGB 299x299 images, this implies a queue size of
# 16GB. If the machine is memory limited, then decrease this factor to
# decrease the CPU memory footprint, accordingly.
flags.DEFINE_integer('input_queue_memory_factor', 16,
                     """Size of the queue of preprocessed images. """
                     """Default is ideal but try smaller values, e.g. """
                     """4, 2 or 1, if host memory is constrained. See """
                     """comments in code for more details.""")
FLAGS = flags.FLAGS


class InputPipeline(object):
  """TFRecord input pipeline."""

  def __init__(self, dataset, is_training, batch_size, data_format):
    self._dataset = dataset
    self._is_training = is_training
    self._batch_size = batch_size
    self._data_format = data_format

  def inputs(self, num_epochs=None, num_preprocess_threads=None, seed=0):
    """Generate batches inputs.

    Args:
      num_epochs: Int. Number of epochs.
      num_preprocess_threads: Int. Total number of preprocessing threads but
        None defaults to FLAGS.num_preprocess_threads.
      seed: Int. Random seed for shuffling data.

    Returns:
      example: Dictioniary. A batched data input dictionary.
    """
    # Force all input processing onto CPU in order to reserve the GPU for
    # the forward inference and back-propagation.
    with tf.device('/cpu:0'):
      example_batch = self.batch_inputs(
          self.batch_size,
          is_training=self.is_training,
          num_epochs=num_epochs,
          num_preprocess_threads=num_preprocess_threads,
          num_readers=FLAGS.num_readers if self.is_training else 1,
          seed=seed)
      if self.data_format == "NCHW":
        example_batch["image"] = tf.transpose(example_batch["image"],
                                              [0, 3, 1, 2])
    return example_batch

  @abstractmethod
  def parse_example_proto(self, example_serialized):
    """Parses an Example proto."""
    pass

  @abstractmethod
  def preprocess_example(self, example, is_training, thread_id=0):
    """Input preprocessing."""
    pass

  def batch_inputs(self,
                   batch_size,
                   is_training,
                   num_preprocess_threads=None,
                   num_epochs=None,
                   num_readers=1,
                   seed=0):
    """Contruct batches of training or evaluation examples from the image dataset.

    Args:
      dataset: instance of Dataset class specifying the dataset.
        See dataset.py for details.
      batch_size: integer
      train: boolean
      num_preprocess_threads: integer, total number of preprocessing threads
      num_readers: integer, number of parallel readers

    Returns:
      images: 4-D float Tensor of a batch of images
      labels: 1-D integer Tensor of [batch_size].

    Raises:
      ValueError: if data is not found
    """
    dataset = self.dataset
    with tf.name_scope('batch_processing'):
      data_files = self.dataset.data_files()
      if data_files is None:
        raise ValueError('No data files found for this dataset')

      # Create filename_queue
      if is_training:
        filename_queue = tf.train.string_input_producer(
            data_files,
            shuffle=True,
            capacity=16,
            num_epochs=num_epochs,
            seed=seed)
      else:
        filename_queue = tf.train.string_input_producer(
            data_files, shuffle=False, capacity=1, num_epochs=num_epochs)
      if num_preprocess_threads is None:
        num_preprocess_threads = FLAGS.num_preprocess_threads

      if num_preprocess_threads % 4:
        raise ValueError('Please make num_preprocess_threads a multiple '
                         'of 4 (%d % 4 != 0).', num_preprocess_threads)

      if num_readers is None:
        num_readers = FLAGS.num_readers

      if num_readers < 1:
        raise ValueError('Please make num_readers at least 1')

      # Approximate number of examples per shard.
      examples_per_shard = dataset.num_examples_per_epoch() // len(data_files)
      # Size the random shuffle queue to balance between good global
      # mixing (more examples) and memory use (fewer examples).
      # 1 image uses 299*299*3*4 bytes = 1MB
      # The default input_queue_memory_factor is 16 implying a shuffling queue
      # size: examples_per_shard * 16 * 1MB = 17.6GB
      min_queue_examples = examples_per_shard * FLAGS.input_queue_memory_factor
      if is_training:
        examples_queue = tf.RandomShuffleQueue(
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples,
            dtypes=[tf.string],
            seed=seed)
      else:
        examples_queue = tf.FIFOQueue(
            capacity=examples_per_shard + 3 * batch_size, dtypes=[tf.string])

      # Create multiple readers to populate the queue of examples.
      if num_readers > 1:
        enqueue_ops = []
        for _ in range(num_readers):
          reader = dataset.reader()
          _, value = reader.read(filename_queue)
          enqueue_ops.append(examples_queue.enqueue([value]))

        tf.train.queue_runner.add_queue_runner(
            tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
        example_serialized = examples_queue.dequeue()
      else:
        reader = dataset.reader()
        _, example_serialized = reader.read(filename_queue)

      examples = []
      for thread_id in range(num_preprocess_threads):
        # Parse a serialized Example proto to extract the image and metadata.
        example = self.parse_example_proto(example_serialized)
        example = self.preprocess_example(example, is_training, thread_id)
        examples.append(example)

      example_batch = tf.train.batch_join(
          examples,
          batch_size=batch_size,
          capacity=2 * num_preprocess_threads * batch_size)

      return example_batch

  @property
  def dataset(self):
    return self._dataset

  @property
  def is_training(self):
    return self._is_training

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def data_format(self):
    return self._data_format
