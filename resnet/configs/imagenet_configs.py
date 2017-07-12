from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from resnet.configs.config_factory import RegisterConfig


@RegisterConfig("resnet-50")
class ResNet50Config(object):

  def __init__(self):
    super(ResNet50Config, self).__init__()
    self.height = 224
    self.width = 224
    self.num_channel = 3
    self.num_residual_units = [3, 4, 6, 3]  # ResNet-50
    self.strides = [1, 2, 2, 2]
    self.activate_before_residual = [True, False, False, False]
    self.init_stride = 2
    self.init_max_pool = True
    self.init_filter = 7
    self.use_bottleneck = True
    self.relu_leakiness = False
    self.filters = [64, 64, 128, 256, 512]
    self.wd = 1e-4
    self.momentum = 0.9
    self.base_learn_rate = 1e-1
    self.max_train_iter = 600000
    self.lr_scheduler = "fixed"
    self.lr_decay_steps = [160000, 320000, 480000]
    self.lr_list = [1e-2, 1e-3, 1e-4]
    self.name = "resnet-50"
    self.model_class = "resnet"
    self.disp_iter = 10
    self.save_iter = 5000
    self.trainval_iter = 1000
    self.valid_iter = 5000
    self.valid_batch_size = 64  ### Use this if necessary.
    self.div255 = True
    self.run_validation = False
    self.num_classes = 1000
    self.batch_size = 256
    self.preprocessor = "inception"  # VGG or Inception.
    self.seed = 1234
    self.optimizer = "mom"
    self.filter_initialization = "normal"


@RegisterConfig("resnet-101")
class ResNet101Config(ResNet50Config):

  def __init__(self):
    super(ResNet101Config, self).__init__()
    self.num_residual_units = [3, 4, 23, 3]  # ResNet-101
    self.batch_size = 256


@RegisterConfig("resnet-152")
class ResNet152Config(ResNet50Config):

  def __init__(self):
    super(ResNet152Config, self).__init__()
    self.num_residual_units = [3, 8, 36, 3]  # ResNet-152
    self.batch_size = 256


@RegisterConfig("revnet-54")
class RevNet54Config(ResNet50Config):

  def __init__(self):
    super(RevNet54Config, self).__init__()
    self.model_class = "revnet"
    self.manual_gradients = True
    self.num_residual_units = [2, 2, 3, 2]  # RevNet-54
    self.filters = [128, 128, 256, 512, 832]


@RegisterConfig("revnet-104")
class RevNet104Config(ResNet101Config):

  def __init__(self):
    super(RevNet104Config, self).__init__()
    self.model_class = "revnet"
    self.manual_gradients = True
    self.num_residual_units = [2, 2, 11, 2]  # RevNet-104
    self.filters = [128, 128, 256, 512, 832]
    self.name = "revnet-104"
