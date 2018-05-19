#!/usr/bin/env python
"""
Train a CNN on CIFAR.
Author: Mengye Ren (mren@cs.toronto.edu)

Usage:
python run_cifar_train.py    --model           [MODEL NAME]          \
                             --config          [CONFIG FILE]         \
                             --env             [ENV FILE]            \
                             --dataset         [DATASET]             \
                             --data_folder     [DATASET FOLDER]      \
                             --validation                            \
                             --no_validation                         \
                             --logs            [LOGS FOLDER]         \
                             --results         [SAVE FOLDER]         \
                             --gpu             [GPU ID]

Flags:
  --model: See resnet/configs/cifar_exp_config.py. Default resnet-32.
  --config: Not using the pre-defined configs above, specify the JSON file
  that contains model configurations.
  --dataset: Dataset name. Available options are: 1) cifar-10 2) cifar-100.
  --data_folder: Path to data folder, default is data/{DATASET}.
  --validation: Evaluating experiments on validation set.
  --no_validation: Evaluating experiments on test set.
  --logs: Path to logs folder, default is logs/default.
  --results: Path to save folder, default is results.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import json
import numpy as np
import os
import six
import tensorflow as tf

from google.protobuf.text_format import Merge, MessageToString
from tqdm import tqdm

from resnet.configs.resnet_model_config_pb2 import ResnetModelConfig
from resnet.data_tfrecord.data_factory import get_data_inputs
from resnet.data_tfrecord.cifar_dataset import Cifar10Dataset
from resnet.data_tfrecord.cifar_input_pipeline import CifarInputPipeline
from resnet.models.resnet_model import ResnetModel
from resnet.models.model_factory import get_model, get_multi_gpu_model
from resnet.utils.logger import get as get_logger
from resnet.utils.gen_id import gen_id

log = get_logger()

flags = tf.flags
flags.DEFINE_string("config", None, "Manually defined config file.")
flags.DEFINE_string("data_root", "./data", "Dataset root.")
flags.DEFINE_string("dataset", "cifar-10", "Dataset name.")
flags.DEFINE_string("id", None, "Experiment ID.")
flags.DEFINE_string("results", "./results/cifar", "Saving folder.")
flags.DEFINE_string("logs", "./logs/public", "Logging folder.")
flags.DEFINE_string("model", "resnet-32-v1", "Model type.")
flags.DEFINE_bool("validation", False, "Whether run validation set.")
flags.DEFINE_bool("restore", False, "Whether restore model.")
flags.DEFINE_integer("num_gpu", 1, "Number of GPUs")
flags.DEFINE_bool("eval", False, "Evaluation only")
FLAGS = flags.FLAGS


def _get_config():
  # Manually set config.
  if FLAGS.config is not None:
    config_file = FLAGS.config
  else:
    if FLAGS.restore:
      save_folder = os.path.realpath(
          os.path.abspath(os.path.join(FLAGS.results, FLAGS.id)))
      config_file = os.path.join(save_folder, "conf.prototxt")
    else:
      config_file = os.path.join('resnet/configs/cifar/{}.prototxt'.format(
          FLAGS.model))
  config = ResnetModelConfig()
  print(config_file)
  Merge(open(config_file).read(), config)
  return config


def _get_model(config,
               inp,
               label,
               bsize,
               is_training,
               name_scope,
               num_replica,
               reuse=None):
  kwargs = {
      "is_training": is_training,
      "inp": inp,
      "label": label,
      "batch_size": bsize
  }
  if num_replica == 1:
    with tf.name_scope(name_scope):
      with tf.variable_scope("Model", reuse=reuse):
        return get_model('resnet', config, **kwargs)
  elif num_replica > 1:
    kwargs["num_replica"] = num_replica
    with tf.name_scope(name_scope):
      with tf.variable_scope("Model", reuse=reuse):
        return get_multi_gpu_model('resnet', config, **kwargs)


def train_step(sess, model):
  """Train step."""
  return model.train_step(sess)


def evaluate(sess, model, num_batches):
  """Runs evaluation."""
  num_correct = 0.0
  count = 0
  for _ in six.moves.xrange(num_batches):
    correct_ = model.eval_step(sess)
    num_correct += np.sum(correct_)
    count += correct_.size
  acc = (num_correct / count)
  return acc


def save(sess, saver, global_step, config, save_folder):
  """Snapshots a model."""
  if not os.path.isdir(save_folder):
    os.makedirs(save_folder)
  config_file = os.path.join(save_folder, "conf.json")
  with open(config_file, "w") as f:
    f.write(MessageToString(config))
  log.info("Saving to {}".format(save_folder))
  saver.save(
      sess, os.path.join(save_folder, "model.ckpt"), global_step=global_step)


def train_model(sess, exp_id, config, model, model_val, save_folder=None):
  """Trains a CIFAR model.

  Args:
      exp_id: String. Experiment ID.
      config: Config object
      train_data: Dataset iterator.
      test_data: Dataset iterator.

  Returns:
      acc: Final test accuracy
  """
  niter_start = int(model.global_step.eval())
  w_list = tf.trainable_variables()
  log.info("Model initialized.")
  num_params = np.array([
      np.prod(np.array([int(ss) for ss in w.get_shape()])) for w in w_list
  ]).sum()
  log.info("Number of parameters {}".format(num_params))
  log.info("Experiment ID {}".format(exp_id))
  it = tqdm(range(niter_start, config.max_train_iter), desc='train', ncols=0)
  trn_acc = 0.0
  val_acc = 0.0
  for niter in it:
    ce = train_step(sess, model)

    if (niter + 1) % 1000 == 0 or niter == 0:
      trn_acc = evaluate(sess, model, 50)
      val_acc = evaluate(sess, model_val, 50)
      print()

    if (niter + 1) % 10 == 0 or niter == 0:
      it.set_postfix(
          ce="{:.3e}".format(ce),
          trn_acc="{:.3f}".format(trn_acc * 100.0),
          val_acc="{:.3f}".format(val_acc * 100.0),
          lr="{:.3e}".format(model.lr.eval()))
  acc = evaluate(sess, model_val, 50)
  return acc


def main():
  # Loads parammeters.
  config = _get_config()
  if FLAGS.id is None:
    dataset_name = FLAGS.dataset
    exp_id = "exp_" + dataset_name + "_" + FLAGS.model
    exp_id = gen_id(exp_id)
  else:
    exp_id = FLAGS.id
    dataset_name = exp_id.split("_")[1]

  # Initializes variables.
  with tf.Graph().as_default(), tf.Session() as sess:
    # Configures dataset objects.
    log.info("Building dataset")
    data_dir = os.path.join(FLAGS.data_root, FLAGS.dataset + "-tf")
    trn_data = get_data_inputs(
        FLAGS.dataset,
        "cifar",
        data_dir,
        "train",
        True,
        batch_size=config.train_batch_size,
        data_format=config.data_format)
    trn_batch = trn_data.inputs()
    val_data = get_data_inputs(
        FLAGS.dataset,
        "cifar",
        data_dir,
        "validation",
        False,
        batch_size=config.eval_batch_size,
        data_format=config.data_format)
    val_batch = val_data.inputs()
    test_data = get_data_inputs(
        FLAGS.dataset,
        "cifar",
        data_dir,
        "test",
        False,
        batch_size=config.eval_batch_size,
        data_format=config.data_format)
    test_batch = test_data.inputs()

    # Set up number of classes.
    config.num_classes = trn_data.dataset.num_classes()

    model = _get_model(
        config,
        trn_batch["image"],
        trn_batch["label"],
        config.train_batch_size,
        True,
        'Train',
        num_replica=FLAGS.num_gpu,
        reuse=None)
    model_val = _get_model(
        config,
        val_batch["image"],
        val_batch["label"],
        config.eval_batch_size,
        False,
        'Val',
        num_replica=FLAGS.num_gpu,
        reuse=True)
    model_test = _get_model(
        config,
        test_batch["image"],
        test_batch["label"],
        config.eval_batch_size,
        False,
        'Test',
        num_replica=FLAGS.num_gpu,
        reuse=True)

    saver = tf.train.Saver()
    if FLAGS.restore:
      log.info("Restore checkpoint {}".format(save_folder))
      saver.restore(sess, tf.train.latest_checkpoint(save_folder))
    else:
      sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    np.random.seed(0)
    if not hasattr(config, "seed"):
      tf.set_random_seed(1234)
      log.info("Setting tensorflow random seed={:d}".format(1234))
    else:
      log.info("Setting tensorflow random seed={:d}".format(config.seed))
      tf.set_random_seed(config.seed)

    # Trains a model.
    if not FLAGS.eval:
      acc = train_model(
          sess, exp_id, config, model, model_val, save_folder=None)

    val_acc, _ = evaluate(sess, model_val, 50)
    test_acc, _ = evaluate(sess, model_test, 100)

  log.info("Final val accuracy = {:.3f}".format(val_acc * 100))
  log.info("Final test accuracy = {:.3f}".format(test_acc * 100))


if __name__ == "__main__":
  main()
