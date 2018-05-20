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
import horovod.tensorflow as hvd

from google.protobuf.text_format import Merge, MessageToString
from tqdm import tqdm

from resnet.configs.resnet_model_config_pb2 import ResnetModelConfig
from resnet.data_tfrecord.data_factory import get_data_inputs
from resnet.data_tfrecord.imagenet_dataset import ImagenetDataset
from resnet.data_tfrecord.imagenet_input_pipeline import ImagenetInputPipeline
from resnet.models.resnet_model import ResnetModel
from resnet.models.model_factory import get_model
from resnet.utils.logger import get as get_logger
from resnet.utils.gen_id import gen_id

log = get_logger()

flags = tf.flags
flags.DEFINE_bool("restore", False, "Restore checkpoint")
flags.DEFINE_integer("max_num_steps", -1, "Maximum number of steps")
flags.DEFINE_integer("num_gpu", 4, "Number of GPUs")
flags.DEFINE_integer("num_pass", 1, "Number of forward-backwad passes")
flags.DEFINE_string("config", None, "Manually defined config file")
flags.DEFINE_string("data_root", "./data", "Dataset root.")
flags.DEFINE_string("id", None, "Experiment ID")
flags.DEFINE_string("model", "resnet-50-v1", "Model name")
flags.DEFINE_string("results", "./results/imagenet", "Saving folder")
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
      config_file = os.path.join(
          'resnet/configs/imagenet/{}.prototxt'.format(FLAGS.model))
  config = ResnetModelConfig()
  print(config_file)
  Merge(open(config_file).read(), config)
  return config


def _get_model(config, inp, label, bsize, is_training):
  """Builds a model."""
  kwargs = {
      "is_training": is_training,
      "inp": inp,
      "label": label,
      "batch_size": bsize,
      "inference_only": True
  }
  with tf.name_scope("Train"):
    with tf.variable_scope("Model", reuse=None):
      with log.verbose_level(2):
        m = get_model("resnet", config, **kwargs)

  global_step = tf.get_variable(
      "global_step", [],
      initializer=tf.constant_initializer(0),
      trainable=False,
      dtype=tf.int64)
  lr = tf.train.piecewise_constant(
      global_step, config.learn_rate_decay_steps,
      [config.learn_rate] + list(config.learn_rate_list))
  m._lr = lr
  m._global_step = global_step
  opt = tf.train.MomentumOptimizer(lr, 0.9)
  opt = hvd.DistributedOptimizer(opt)
  hooks = [hvd.BroadcastGlobalVariablesHook(0)]
  m._train_op = opt.minimize(m.cost, global_step=global_step, name="train_step")
  tf.summary.scalar("train ce", m.cross_ent)
  return m, hooks


def train_step(sess, model):
  """Train step."""
  ce = model.train_step(sess)
  return ce


def save(sess, saver, global_step, config, save_folder):
  """Snapshots a model."""
  if not os.path.isdir(save_folder):
    os.makedirs(save_folder)
  config_file = os.path.join(save_folder, "conf.prototxt")
  with open(config_file, "w") as f:
    f.write(MessageToString(config))
  log.info("Saving to {}".format(save_folder))
  saver.save(
      sess, os.path.join(save_folder, "model.ckpt"), global_step=global_step)


def _get_exp_logger(sess, log_folder):
  """Gets a TensorBoard logger."""
  with tf.name_scope('Summary'):
    writer = tf.summary.FileWriter(log_folder)

  class ExperimentLogger():

    def log(self, niter, name, value):
      """Logs training cross entropy."""
      summary = tf.Summary()
      summary.value.add(tag=name, simple_value=value)
      writer.add_summary(summary, niter)

    def flush(self):
      """Flushes results to disk."""
      writer.flush()

    def close(self):
      """Closes writer."""
      writer.close()

  return ExperimentLogger()


def train_model(sess, exp_id, config, model, save_folder=None):
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
  log.info("Config: {}".format(MessageToString(config)))
  exp_logger = _get_exp_logger(sess, save_folder)

  saver = tf.train.Saver(max_to_keep=None)
  if FLAGS.restore:
    log.info("Restore checkpoint \"{}\"".format(save_folder))
    saver.restore(sess, tf.train.latest_checkpoint(save_folder))
  else:
    sess.run(tf.global_variables_initializer())

  # Start the queue runners.
  coord = tf.train.Coordinator()
  tf.train.start_queue_runners(sess=sess, coord=coord)

  # Count parameters.
  w_list = tf.trainable_variables()
  num_params = np.array(
      [np.prod(np.array([int(ss) for ss in w.get_shape()]))
       for w in w_list]).sum()
  log.info("Number of parameters {}".format(num_params))

  max_train_iter = config.max_train_iter
  niter_start = int(model.global_step.eval())

  # Add upper bound to the number of steps.
  if FLAGS.max_num_steps > 0:
    max_train_iter = min(max_train_iter, niter_start + FLAGS.max_num_steps)

  log.info("Experiment ID {}".format(exp_id))
  it = tqdm(range(niter_start, config.max_train_iter), desc="train", ncols=0)
  for niter in it:
    ce = train_step(sess, model)

    if (niter + 1) % 10 == 0 or niter == 0:
      exp_logger.log(niter + 1, "train ce", ce)
      exp_logger.flush()
      it.set_postfix(ce="{:.3e}".format(ce))

    if (niter + 1) % 5000 == 0 or niter == 0:
      if save_folder is not None:
        save(sess, saver, model.global_step, config, save_folder)
      exp_logger.log(niter + 1, "learn rate", model.lr.eval())
      exp_logger.flush()
  coord.request_stop()
  coord.join(threads)
  exp_logger.close()


def main():
  # Loads parammeters.
  config = _get_config()
  hvd.init()

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

  sconfig = tf.ConfigProto()
  with tf.Graph().as_default():
    np.random.seed(0)
    tf.set_random_seed(1234)

    # Configures dataset objects.
    log.info("Building dataset")
    trn_data = get_data_inputs(
        DATASET,
        DATASET,
        os.path.join(FLAGS.data_root, DATASET),
        "train",
        True,
        batch_size=config.train_batch_size,
        data_format=config.data_format)
    trn_batch = trn_data.inputs(seed=hvd.local_rank() * 1000)
    # rnd = np.random.RandomState(hvd.local_rank() * 1000)
    # trn_batch = {
    #     "image":
    #     tf.constant(
    #         np.random.uniform(-1, 1, [64, 3, 224, 224]).astype(np.float32)),
    #     "label":
    #     tf.constant(np.random.uniform(0, 10, [64]).astype(np.int32))
    # }

    # Builds models.
    log.info("Building models")
    model, hooks = _get_model(
        config,
        trn_batch["image"],
        tf.squeeze(trn_batch["label"]),
        config.train_batch_size,
        is_training=True)

    # Count parameters.
    w_list = tf.trainable_variables()
    num_params = np.array(
        [np.prod(np.array([int(ss) for ss in w.get_shape()]))
         for w in w_list]).sum()
    log.info("Number of parameters {}".format(num_params))

    sconfig = tf.ConfigProto()
    sconfig.gpu_options.visible_device_list = str(hvd.local_rank())
    sconfig.gpu_options.allow_growth = True
    summary_op = tf.summary.merge_all()
    summary_hook = tf.train.SummarySaverHook(
        save_steps=10, output_dir=save_folder, summary_op=summary_op)

    log.info("Saving to {}".format(save_folder))
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=save_folder,
        config=sconfig,
        hooks=hooks + [summary_hook]) as mon_sess:

      log.info("Config: {}".format(MessageToString(config)))
      if str(hvd.local_rank()) == "0":
        exp_logger = _get_exp_logger(mon_sess, save_folder)

      max_train_iter = config.max_train_iter
      niter_start = int(mon_sess.run(model.global_step))

      # Add upper bound to the number of steps.
      if FLAGS.max_num_steps > 0:
        max_train_iter = min(max_train_iter, niter_start + FLAGS.max_num_steps)

      log.info("Experiment ID {}".format(exp_id))
      it = tqdm(
          range(niter_start, config.max_train_iter), desc="train", ncols=0)
      for niter in it:
        # Perform synchronous training.
        ce, _ = mon_sess.run([model.cross_ent, model.train_op])
        if ((niter + 1) % 10 == 0 or
            niter == 0) and str(hvd.local_rank()) == "0":
          exp_logger.log(niter + 1, "train ce", ce)
          exp_logger.flush()
          it.set_postfix(ce="{:.3e}".format(ce))


if __name__ == "__main__":
  main()
