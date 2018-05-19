#! /usr/bin/env python
"""
Converts CIFAR to TFRecord.
Author: Mengye Ren (mren@cs.toronto.edu)

Output TFRecord in the following location:
[data folder]/[dataset name]-tf/train-00000-of-00005
[data folder]/[dataset name]-tf/train-00001-of-00005
...
[data folder]/[dataset name]-tf/train-00004-of-00005
[data folder]/[dataset name]-tf/validation-00000-of-00001

Usage:
./build_cifar_data.py --dataset      [DATASET NAME]            \
                      --data_folder  [DATA FOLDER PATH]
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import pickle as pkl
import six
import tensorflow as tf

flags = tf.flags
flags.DEFINE_string("data_folder", "./data/cifar-10", "Data set folder")
flags.DEFINE_string("dataset", "cifar-10", "Data set name")
flags.DEFINE_string("output_folder", "./data/cifar-10-tf",
                    "TFRecord output folder")
flags.DEFINE_integer("num_val", 5000, "Number of validation data, default 5000")
flags.DEFINE_integer("seed", 0, "Random seed")
FLAGS = tf.flags.FLAGS


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(image, label):
  example = tf.train.Example(
      features=tf.train.Features(feature={
          'image': _bytes_feature(image),
          'label': _int64_feature(label)
      }))

  return example


def _unpickle(filename):
  try:
    with open(filename, 'rb') as fo:
      data_dict = pkl.load(fo)
  except:
    with open(filename, 'rb') as fo:
      data_dict = pkl.load(fo, encoding="bytes")
  return data_dict


def _split(num, seed, partitions):
  all_idx = np.arange(num)
  rnd = np.random.RandomState(seed)
  rnd.shuffle(all_idx)
  siz = 0
  results = []
  for pp in partitions:
    results.append(all_idx[siz:siz + pp])
    siz += pp
  return results


def read_cifar_10(data_folder):
  train_file_list = [
      "data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4",
      "data_batch_5"
  ]
  test_file_list = ["test_batch"]
  data_dict = {}
  for file_list, name in zip([train_file_list, test_file_list],
                             ["train", "validation"]):
    img_list = []
    label_list = []
    for ii in six.moves.xrange(len(file_list)):
      data_dict = _unpickle(
          os.path.join(data_folder, "cifar-10-batches-py", file_list[ii]))
      _img = data_dict[b"data"]
      _label = data_dict[b"labels"]
      _img = _img.reshape([-1, 3, 32, 32])
      _img = _img.transpose([0, 2, 3, 1])
      img_list.append(_img)
      label_list.append(_label)
    img = np.concatenate(img_list, axis=0)
    label = np.concatenate(label_list, axis=0)
    if name == "train":
      train_img = img
      train_label = label
    else:
      test_img = img
      test_label = label
  return train_img, train_label, test_img, test_label


def read_cifar_100(data_folder):
  train_file_list = ["train"]
  test_file_list = ["test"]
  data_dict = _unpickle(
      os.path.join(data_folder, "cifar-100-python", train_file_list[0]))
  train_img = data_dict[b"data"]
  train_label = np.array(data_dict[b"fine_labels"])

  data_dict = _unpickle(
      os.path.join(data_folder, "cifar-100-python", test_file_list[0]))
  test_img = data_dict[b"data"]
  test_label = np.array(data_dict[b"fine_labels"])

  train_img = train_img.reshape([-1, 3, 32, 32])
  train_img = train_img.transpose([0, 2, 3, 1])
  test_img = test_img.reshape([-1, 3, 32, 32])
  test_img = test_img.transpose([0, 2, 3, 1])
  return train_img, train_label, test_img, test_label


def serialize_to_tf_record(basename, num_shard, images, labels):
  output_filename = basename + "-{:05d}-of-{:05d}"
  num_example = images.shape[0]
  num_example_per_shard = int(np.ceil(num_example / float(num_shard)))
  for ii in six.moves.xrange(num_shard):
    _filename = output_filename.format(ii, num_shard)
    with tf.python_io.TFRecordWriter(_filename) as writer:
      start = num_example_per_shard * ii
      end = min(num_example_per_shard * (ii + 1), num_example)
      for jj in six.moves.xrange(start, end):
        _example = _convert_to_example(images[jj].tobytes(), labels[jj])
        writer.write(_example.SerializeToString())
      writer.flush()


def trainval_split(img, label, num_val, seed):
  assert img.shape[0] == label.shape[0]
  num = img.shape[0]
  trainval_partition = [num - num_val, num_val]
  idx = _split(num, seed, trainval_partition)
  return img[idx[0]], label[idx[0]], img[idx[1]], label[idx[1]]


def convert_cifar(dataset, data_folder, num_val, output_folder, seed):
  if dataset == "cifar-10":
    train_img, train_label, test_img, test_label = read_cifar_10(data_folder)
  elif dataset == "cifar-100":
    train_img, train_label, test_img, test_label = read_cifar_100(data_folder)
  else:
    raise ValueError("Unknown dataset {}".format(dataset))

  if not os.path.exists(output_folder):
    os.makedirs(output_folder)

  serialize_to_tf_record(
      os.path.join(output_folder, "trainval"), 5, train_img, train_label)

  train_img, train_label, val_img, val_label = trainval_split(
      train_img, train_label, num_val, seed)

  serialize_to_tf_record(
      os.path.join(output_folder, "train"), 4, train_img, train_label)

  serialize_to_tf_record(
      os.path.join(output_folder, "validation"), 1, val_img, val_label)

  serialize_to_tf_record(
      os.path.join(output_folder, "test"), 1, test_img, test_label)


def main():
  convert_cifar(FLAGS.dataset, FLAGS.data_folder, FLAGS.num_val,
                FLAGS.output_folder, FLAGS.seed)


if __name__ == "__main__":
  main()
