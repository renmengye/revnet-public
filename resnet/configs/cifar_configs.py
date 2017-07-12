from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from resnet.configs.config_factory import RegisterConfig


@RegisterConfig("resnet-32")
class ResNet32Config(object):

  def __init__(self):
    super(ResNet32Config, self).__init__()
    self.batch_size = 100
    self.height = 32
    self.width = 32
    self.num_channel = 3
    self.min_lrn_rate = 0.0001
    self.base_learn_rate = 1e-1
    self.num_residual_units = [5, 5, 5]  # ResNet-32
    self.seed = 1234
    self.strides = [1, 2, 2]
    self.activate_before_residual = [True, False, False]
    self.init_stride = 1
    self.init_max_pool = False
    self.init_filter = 3
    self.use_bottleneck = False
    self.relu_leakiness = False
    self.filters = [16, 16, 32, 64]
    self.wd = 2e-4
    self.optimizer = "mom"
    self.max_train_iter = 80000
    self.lr_decay_steps = [40000, 60000]
    self.lr_scheduler_type = "fixed"
    self.lr_list = [1e-2, 1e-3]
    self.momentum = 0.9
    self.name = "resnet-32"
    self.model_class = "resnet"
    self.filter_initialization = "normal"
    self.disp_iter = 100
    self.save_iter = 10000
    self.valid_iter = 1000
    self.prefetch = True
    self.data_aug = True
    self.whiten = False  # Original TF has whiten.
    self.div255 = True
    self.seed = 0
    self.num_classes = None


@RegisterConfig("resnet-110")
class ResNet110Config(ResNet32Config):

  def __init__(self):
    super(ResNet110Config, self).__init__()
    self.num_residual_units = [18, 18, 18]  # ResNet-110


@RegisterConfig("resnet-164")
class ResNet164Config(ResNet32Config):

  def __init__(self):
    super(ResNet164Config, self).__init__()
    self.num_residual_units = [18, 18, 18]  # ResNet-164
    self.use_bottleneck = True


@RegisterConfig("revnet-38")
class RevNet38Config(ResNet32Config):

  def __init__(self):
    super(RevNet38Config, self).__init__()
    self.model_class = "revnet"
    self.manual_gradients = True
    self.filters = [32, 32, 64, 112]
    self.num_residual_units = [3, 3, 3]


@RegisterConfig("revnet-110")
class RevNet110Config(ResNet110Config):

  def __init__(self):
    super(RevNet110Config, self).__init__()
    self.model_class = "revnet"
    self.manual_gradients = True
    self.filters = [32, 32, 64, 128]
    self.num_residual_units = [9, 9, 9]


@RegisterConfig("revnet-164")
class RevNet164Config(ResNet164Config):

  def __init__(self):
    super(RevNet164Config, self).__init__()
    self.model_class = "revnet"
    self.manual_gradients = True
    self.filters = [32, 32, 64, 128]
    self.num_residual_units = [9, 9, 9]
