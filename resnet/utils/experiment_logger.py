from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime
import os
import sys

from resnet.utils import logger

log = logger.get()


class ExperimentLogger():
  """Writes experimental logs to CSV file."""

  def __init__(self, logs_folder):
    """Initialize files."""
    self._write_to_csv = logs_folder is not None
    self.logs_folder = logs_folder

    if self._write_to_csv:
      if not os.path.isdir(logs_folder):
        os.makedirs(logs_folder)

      catalog_file = os.path.join(logs_folder, "catalog")
      self.catalog_file = catalog_file

      with open(catalog_file, "w") as f:
        f.write("filename,type,name\n")

      with open(catalog_file, "a") as f:
        f.write("{},plain,{}\n".format("cmd.txt", "Commands"))

      with open(os.path.join(logs_folder, "cmd.txt"), "w") as f:
        f.write(" ".join(sys.argv))

      with open(catalog_file, "a") as f:
        f.write("train_ce.csv,csv,Train Loss (Cross Entropy)\n")
        f.write("train_acc.csv,csv,Train Accuracy\n")
        f.write("valid_acc.csv,csv,Validation Accuracy\n")
        f.write("learn_rate.csv,csv,Learning Rate\n")

      self.train_file_name = os.path.join(logs_folder, "train_ce.csv")
      if not os.path.exists(self.train_file_name):
        with open(self.train_file_name, "w") as f:
          f.write("step,time,ce\n")

      self.trainval_file_name = os.path.join(logs_folder, "train_acc.csv")
      if not os.path.exists(self.trainval_file_name):
        with open(self.trainval_file_name, "w") as f:
          f.write("step,time,acc\n")

      self.val_file_name = os.path.join(logs_folder, "valid_acc.csv")
      if not os.path.exists(self.val_file_name):
        with open(self.val_file_name, "w") as f:
          f.write("step,time,acc\n")

      self.lr_file_name = os.path.join(logs_folder, "learn_rate.csv")
      if not os.path.exists(self.lr_file_name):
        with open(self.lr_file_name, "w") as f:
          f.write("step,time,lr\n")

  def log_value(self, niter, key, value, name, name_short=None):
    if name_short is None:
      name_short = key
    log.info("{} = {:.4e}".format(name, value))
    if self._write_to_csv:
      file_name_short = name_short + ".csv"
      file_name = os.path.join(self.logs_folder, file_name_short)
      if not os.path.exists(file_name):
        with open(file_name, "w") as f:
          f.write("step,time,{}\n".format(key))
        with open(self.catalog_file, "a") as f:
          f.write("{},csv,{}\n".format(file_name_short, name))
      with open(file_name, "a") as f:
        f.write("{:d},{:s},{:e}\n".format(
            niter + 1, datetime.datetime.now().isoformat(), value))

  def log_value_list(self, niter, keys, values, name, name_short):
    if self._write_to_csv:
      file_name_short = name_short + ".csv"
      file_name = os.path.join(self.logs_folder, file_name_short)
      if not os.path.exists(file_name):
        with open(file_name, "w") as f:
          f.write("step,time,{}\n".format(",".join(keys)))
        with open(self.catalog_file, "a") as f:
          f.write("{},csv,{}\n".format(file_name_short, name))
      with open(file_name, "a") as f:
        f.write("{:d},{:s},{}\n".format(
            niter + 1,
            datetime.datetime.now().isoformat(), ",".join(
                ["{:e}".format(v) for v in values])))

  def log_train_ce(self, niter, ce):
    """Writes training CE."""
    log.info("Train Step = {:06d} || CE loss = {:.4e}".format(niter + 1, ce))
    if self._write_to_csv:
      with open(self.train_file_name, "a") as f:
        f.write("{:d},{:s},{:e}\n".format(
            niter + 1, datetime.datetime.now().isoformat(), ce))

  def log_train_acc(self, niter, acc):
    """Writes training accuracy."""
    log.info("Train accuracy = {:.3f}".format(acc * 100))
    if self._write_to_csv:
      with open(self.trainval_file_name, "a") as f:
        f.write("{:d},{:s},{:e}\n".format(
            niter + 1, datetime.datetime.now().isoformat(), acc))

  def log_valid_acc(self, niter, acc):
    """Writes validation accuracy."""
    log.info("Valid accuracy = {:.3f}".format(acc * 100))
    if self._write_to_csv:
      with open(self.val_file_name, "a") as f:
        f.write("{:d},{:s},{:e}\n".format(
            niter + 1, datetime.datetime.now().isoformat(), acc))

  def log_learn_rate(self, niter, lr):
    """Writes validation accuracy."""
    if self._write_to_csv:
      with open(self.lr_file_name, "a") as f:
        f.write("{:d},{:s},{:e}\n".format(
            niter + 1, datetime.datetime.now().isoformat(), lr))
