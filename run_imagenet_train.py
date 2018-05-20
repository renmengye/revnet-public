#!/usr/bin/env python
"""
Trains a CNN on ImageNet, across multiple GPUs in a single node.
Author: Mengye Ren (mren@cs.toronto.edu)

Usage:
./run_imagenet_train.py --model           [MODEL NAME]                     \
                        --config          [CONFIG FILE]                    \
                        --id              [EXPERIMENT ID]                  \
                        --logs            [LOGS FOLDER]                    \
                        --results         [SAVE FOLDER]                    \
                        --restore                                          \
                        --norestore                                        \
                        --max_num_steps   [MAX NUM OF STEPS FOR THIS RUN]  \
                        --num_gpu         [NUMBER OF GPU]                  \
                        --num_pass        [NUMBER OF FW/BW PASS]

Flags:
  --model: Model type. Available options are:
       1) resnet-50
       2) resnet-101
  --id: Experiment ID, optional for new experiment.
  --config: Not using the pre-defined configs above, specify the JSON file
    that contains model configurations.
  --logs: Path to logs folder, default is ./logs/default.
  --results: Path to save folder, default is ./results/imagenet.
  --restore: Whether or not to restore checkpoint. Checkpoint should be
    present in [SAVE FOLDER]/[EXPERIMENT ID] folder.
  --max_num_steps: Maximum number of steps for this training session.
  --num_gpu: Number of GPU to perform data parallelism.
  --num_pass: Number of forward and backward passes to average gradients.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import os
import tensorflow as tf

from google.protobuf.text_format import Merge, MessageToString
from tqdm import tqdm

from resnet.configs import get_config, get_config_from_json
from resnet.data_tfrecord import get_data_inputs
from resnet.models.model_factory import get_multi_gpu_model
from resnet.utils import (ExperimentLogger, FixedLearnRateScheduler)
from resnet.utils import logger, gen_id

log = logger.get()

flags = tf.flags
flags.DEFINE_string("config", None, "Manually defined config file")
flags.DEFINE_string("id", None, "Experiment ID")
flags.DEFINE_string("results", "./results/imagenet", "Saving folder")
flags.DEFINE_string("logs", "./logs/public", "Logging folder")
flags.DEFINE_string("model", "resnet-50", "Model name")
flags.DEFINE_bool("restore", False, "Restore checkpoint")
flags.DEFINE_integer("max_num_steps", -1, "Maximum number of steps")
flags.DEFINE_integer("num_gpu", 4, "Number of GPUs")
flags.DEFINE_integer("num_pass", 1, "Number of forward-backwad passes")
FLAGS = flags.FLAGS

DATASET = "imagenet"


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
      config_file = os.path.join('resnet/configs/imagenet/{}.prototxt'.format(
          FLAGS.model))
  config = ResnetModelConfig()
  print(config_file)
  Merge(open(config_file).read(), config)
  return config


def _get_model(config, inp, label, num_replica, num_pass, is_training):
  """Builds a model."""
  with tf.name_scope("Train"):
    with tf.variable_scope("Model", reuse=None):
      with log.verbose_level(2):
        return get_multi_gpu_model(
            config.model_class,
            config,
            is_training=is_training,
            num_replica=num_replica,
            num_pass=num_pass,
            inp=inp,
            label=label,
            batch_size=config.batch_size)


def _get_dataset(config):
  """Prepares a dataset input tensors."""
  num_preprocess_threads = FLAGS.num_preprocess_threads * FLAGS.num_gpu
  trn_batch = get_data_inputs(
      DATASET, FLAGS.data_dir, "train", True, config.batch_size,
      DATASET).inputs(num_preprocess_threads=num_preprocess_threads)
  return trn_batch['image'], trn_batch['label']


def train_step(sess, model):
  """Train step."""
  ce = model.train_step(sess)
  return ce


def save(sess, saver, global_step, config, save_folder):
  """Snapshots a model."""
  if not os.path.isdir(save_folder):
    os.makedirs(save_folder)
  config_file = os.path.join(save_folder, "conf.json")
  with open(config_file, "w") as f:
    f.write(json.dumps(config, default=lambda o: o.__dict__))
  log.info("Saving to {}".format(save_folder))
  saver.save(
      sess, os.path.join(save_folder, "model.ckpt"), global_step=global_step)


def train_model(exp_id, config, model, save_folder=None, logs_folder=None):
  """Trains an ImageNet model.

  Args:
    exp_id: String. Experiment ID.
    config: Config object.
    model: Model object.
    save_folder: Folder to save all checkpoints.
    logs_folder: Folder to save all training logs.

  Returns:
    acc: Final test accuracy
  """
  log.info("Config: {}".format(config.__dict__))
  exp_logger = ExperimentLogger(logs_folder)

  with tf.Session(
      config=tf.ConfigProto(allow_soft_placement=True,
                            allow_groth=True)) as sess:

    found_old_name = False
    if FLAGS.restore:
      ckpt = tf.train.latest_checkpoint(save_folder)
      from tensorflow.python import pywrap_tensorflow
      reader = pywrap_tensorflow.NewCheckpointReader(ckpt)
      var_to_shape_map = reader.get_variable_to_shape_map()
      for key in sorted(var_to_shape_map):
        if key == "Train/Model/learn_rate":
          found_old_name = True
          break

    # A hack to load compatible models.
    if found_old_name:
      variables = tf.global_variables()
      names = map(lambda x: x.name, variables)
      names = map(lambda x: x.strip(":0"), names)
      names = map(
          lambda x: x.replace("Model/learn_rate", "Train/Model/learn_rate"),
          names)
      var_dict = dict(zip(names, variables))
    else:
      var_dict = None

    ### Keep all checkpoints here!
    #saver = tf.train.Saver(max_to_keep=None)
    saver = tf.train.Saver(var_dict, max_to_keep=None)
    if FLAGS.restore:
      log.info("Restore checkpoint \"{}\"".format(save_folder))
      saver.restore(sess, tf.train.latest_checkpoint(save_folder))
    else:
      sess.run(tf.global_variables_initializer())

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    # Count parameters.
    w_list = tf.trainable_variables()
    num_params = np.array([
        np.prod(np.array([int(ss) for ss in w.get_shape()])) for w in w_list
    ]).sum()
    log.info("Number of parameters {}".format(num_params))

    max_train_iter = config.max_train_iter
    niter_start = int(model.global_step.eval())

    # Add upper bound to the number of steps.
    if FLAGS.max_num_steps > 0:
      max_train_iter = min(max_train_iter, niter_start + FLAGS.max_num_steps)

    # Set up learning rate schedule.
    if config.lr_scheduler == "fixed":
      lr_scheduler = FixedLearnRateScheduler(
          sess,
          model,
          config.base_learn_rate,
          config.lr_decay_steps,
          lr_list=config.lr_list)
    else:
      raise Exception("Unknown learning rate scheduler {}".format(
          config.lr_scheduler))

    for niter in tqdm(range(niter_start, config.max_train_iter), desc=exp_id):
      lr_scheduler.step(niter)
      ce = train_step(sess, model)

      if (niter + 1) % config.disp_iter == 0 or niter == 0:
        exp_logger.log_train_ce(niter, ce)

      if (niter + 1) % config.save_iter == 0 or niter == 0:
        if save_folder is not None:
          save(sess, saver, model.global_step, config, save_folder)
        exp_logger.log_learn_rate(niter, model.lr.eval())


def main():
  # Loads parammeters.
  config = _get_config()

  if FLAGS.id is None:
    exp_id = "exp_" + DATASET + "_" + FLAGS.model
    exp_id = gen_id(exp_id)
  else:
    exp_id = FLAGS.id

  if FLAGS.results is not None:
    save_folder = os.path.realpath(
        os.path.abspath(os.path.join(FLAGS.results, exp_id)))
    if not os.path.exists(save_folder):
      os.makedirs(save_folder)
  else:
    save_folder = None

  if FLAGS.logs is not None:
    logs_folder = os.path.realpath(
        os.path.abspath(os.path.join(FLAGS.logs, exp_id)))
    if not os.path.exists(logs_folder):
      os.makedirs(logs_folder)
  else:
    logs_folder = None

  # Initializes variables.
  with tf.Graph().as_default():
    np.random.seed(0)
    tf.set_random_seed(1234)

    # Configures dataset objects.
    log.info("Building dataset")
    inp, label = _get_dataset(config)

    # Builds models.
    log.info("Building models")
    model = _get_model(
        config,
        inp,
        label,
        num_replica=FLAGS.num_gpu,
        num_pass=FLAGS.num_pass,
        is_training=True)

    # Trains a model.
    train_model(
        exp_id, config, model, save_folder=save_folder, logs_folder=logs_folder)


if __name__ == "__main__":
  main()
