from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from resnet.configs.config_factory import RegisterConfig
from resnet.configs.cifar_configs import (ResNet32Config, RevNet38Config)


@RegisterConfig("resnet-test")
class ResNetTestConfig(ResNet32Config):

  def __init__(self):
    super(ResNetTestConfig, self).__init__()
    self.batch_size = 10
    self.num_residual_units = [2, 2, 2]
    self.filters = [2, 2, 4, 8]
    self.num_classes = 10


@RegisterConfig("revnet-test")
class RevNetTestConfig(RevNet38Config):

  def __init__(self):
    super(RevNetTestConfig, self).__init__()
    self.batch_size = 10
    self.num_residual_units = [2, 2]
    self.filters = [16, 16, 32]
    self.height = 8
    self.width = 8
    self.model_class = "revnet"
    self.num_classes = 10


@RegisterConfig("revnet-btl-test")
class RevNetBottleneckTestConfig(RevNet38Config):

  def __init__(self):
    super(RevNetBottleneckTestConfig, self).__init__()
    self.batch_size = 10
    self.num_residual_units = [2, 2]
    self.filters = [16, 16, 32]
    self.height = 8
    self.width = 8
    self.model_class = "revnet"
    self.use_bottleneck = True
    self.num_classes = 10
