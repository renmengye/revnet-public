#!/usr/bin/env python
"""
Evaluates a CNN on ImageNet.
Author: Mengye Ren (mren@cs.toronto.edu)

Usage:
./run_imagenet_eval.py --id              [EXPERIMENT ID]     \
                       --logs            [LOGS FOLDER]       \
                       --results         [SAVE FOLDER]

Flags:
  --id: Experiment ID, optional for new experiment.
  --logs: Path to logs folder, default is ./logs/public.
  --results: Path to save folder, default is ./results/imagenet.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import os
import tensorflow as tf

from tqdm import tqdm

from resnet.configs.config_factory import get_config_from_json
from resnet.data_tfrecord.imagenet_data import ImagenetData
from resnet.data_tfrecord.image_processing import inputs
from resnet.models import get_model
from resnet.utils import logger, ExperimentLogger

flags = tf.flags
flags.DEFINE_string("id", None, "Experiment ID")
flags.DEFINE_string("results", "./results/imagenet", "Saving folder")
flags.DEFINE_string("logs", "./logs/public", "Logging folder")
flags.DEFINE_integer("ckpt_num", -1, "Checkpoint step number")
FLAGS = tf.flags.FLAGS
log = logger.get()

NUM_GPU = 1
NUM_VALID = 50000
BSIZE = 50
NUM_BATCH = NUM_VALID // BSIZE


def _get_config():
  save_folder = os.path.join(FLAGS.results, FLAGS.id)
  return get_config_from_json(os.path.join(save_folder, "conf.json"))


def _get_model(config, trn_inp, trn_label, val_inp, val_label):
  with log.verbose_level(2):
    with tf.name_scope("Train"):
      with tf.variable_scope("Model"):
        m = get_model(
            config.model_class,
            config,
            inp=trn_inp,
            label=trn_label,
            is_training=False,
            inference_only=True)
    with tf.name_scope("Valid"):
      with tf.variable_scope("Model", reuse=True):
        mvalid = get_model(
            config.model_class,
            config,
            inp=val_inp,
            label=val_label,
            is_training=False,
            inference_only=True)
  return m, mvalid


def _get_dataset(config, split):
  """Prepares a dataset input tensors."""
  num_preprocess_threads = FLAGS.num_preprocess_threads * NUM_GPU
  dataset = ImagenetData(subset=split)
  images, labels = inputs(
      dataset,
      cycle=True,
      batch_size=BSIZE,
      num_preprocess_threads=num_preprocess_threads)
  return images, labels


def evaluate(sess, model, num_batch=100):
  """Runs evaluation."""
  num_correct = 0.0
  count = 0
  for bidx in tqdm(range(num_batch)):
    correct = model.eval_step(sess)
    num_correct += np.sum(correct)
    count += correct.size
  acc = (num_correct / count)
  return acc


def eval_model(config,
               trn_model,
               val_model,
               save_folder,
               logs_folder=None,
               ckpt_num=-1):
  log.info("Config: {}".format(config.__dict__))
  exp_logger = ExperimentLogger(logs_folder)
  # Initializes variables.
  with tf.Session() as sess:
    # Start the queue runners.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    saver = tf.train.Saver()
    if ckpt_num == -1:
      ckpt = tf.train.latest_checkpoint(save_folder)
    elif ckpt_num >= 0:
      ckpt = os.path.join(save_folder, "model.ckpt-{}".format(ckpt_num))
    else:
      raise ValueError("Invalid checkpoint number {}".format(ckpt_num))
    log.info("Restoring from {}".format(ckpt))
    if not os.path.exists(ckpt + ".meta"):
      raise ValueError("Checkpoint not exists")
    saver.restore(sess, ckpt)
    train_acc = evaluate(sess, trn_model, num_batch=100)
    val_acc = evaluate(sess, val_model, num_batch=NUM_BATCH)
    niter = int(ckpt.split("-")[-1])
    exp_logger.log_train_acc(niter, train_acc)
    exp_logger.log_valid_acc(niter, val_acc)

    # Stop queues.
    coord.request_stop()
    coord.join(threads)
  return val_acc


def main():
  config = _get_config()
  exp_id = FLAGS.id

  save_folder = os.path.realpath(
      os.path.abspath(os.path.join(FLAGS.results, exp_id)))

  if FLAGS.logs is not None:
    logs_folder = os.path.realpath(
        os.path.abspath(os.path.join(FLAGS.logs, exp_id)))
    if not os.path.exists(logs_folder):
      os.makedirs(logs_folder)
  else:
    logs_folder = None

  # Evaluates a model.
  with tf.Graph().as_default():
    np.random.seed(0)
    tf.set_random_seed(1234)

    # Configures dataset objects.
    log.info("Building dataset")
    trn_inp, trn_label = _get_dataset(config, "train")
    val_inp, val_label = _get_dataset(config, "validation")

    # Builds models.
    log.info("Building models")
    trn_model, val_model = _get_model(config, trn_inp, trn_label, val_inp,
                                      val_label)
    eval_model(
        config,
        trn_model,
        val_model,
        save_folder,
        logs_folder,
        ckpt_num=FLAGS.ckpt_num)


if __name__ == "__main__":
  main()
