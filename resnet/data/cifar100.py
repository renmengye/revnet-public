from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from resnet.data import cifar_input
from resnet.utils import logger

log = logger.get()


class CIFAR100Dataset():

  def __init__(self,
               folder,
               split,
               num_fold=10,
               fold_id=0,
               data_aug=False,
               whiten=False,
               div255=False):
    self.split = split
    self.data = cifar_input.read_CIFAR100(folder)
    num_ex = 50000
    self.split_idx = np.arange(num_ex)
    rnd = np.random.RandomState(0)
    rnd.shuffle(self.split_idx)
    num_valid = int(np.ceil(num_ex / num_fold))
    valid_start = fold_id * num_valid
    valid_end = min((fold_id + 1) * num_valid, num_ex)
    self.valid_split_idx = self.split_idx[valid_start:valid_end]
    self.train_split_idx = np.concatenate(
        [self.split_idx[:valid_start], self.split_idx[valid_end:]])
    if data_aug or whiten:
      with tf.device("/cpu:0"):
        self.inp_preproc, self.out_preproc = cifar_input.cifar_tf_preprocess(
            random_crop=data_aug, random_flip=data_aug, whiten=whiten)
      self.session = tf.Session()
    self.data_aug = data_aug
    self.whiten = whiten
    self.div255 = div255
    if div255 and whiten:
      log.fatal("Applying both /255 and whitening is not recommended.")

  def get_size(self):
    if self.split == "train":
      return 50000
    elif self.split == "traintrain":
      return 45000
    elif self.split == "trainval":
      return 5000
    else:
      return 10000

  def get_batch_idx(self, idx):
    if self.split == "train":
      result = {
          "img": self.data["train_img"][idx],
          "label": self.data["train_label"][idx]
      }
    elif self.split == "traintrain":
      result = {
          "img": self.data["train_img"][self.train_split_idx[idx]],
          "label": self.data["train_label"][self.train_split_idx[idx]]
      }
    elif self.split == "trainval":
      result = {
          "img": self.data["train_img"][self.valid_split_idx[idx]],
          "label": self.data["train_label"][self.valid_split_idx[idx]]
      }
    else:
      result = {
          "img": self.data["test_img"][idx],
          "label": self.data["test_label"][idx]
      }

    if self.data_aug or self.whiten:
      img = np.zeros(result["img"].shape)
      for ii in range(len(idx)):
        img[ii] = self.session.run(
            self.out_preproc, feed_dict={self.inp_preproc: result["img"][ii]})
      result["img"] = img
    if self.div255:
      result["img"] = result["img"] / 255.0
    return result
