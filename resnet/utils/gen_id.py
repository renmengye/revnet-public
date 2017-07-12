import numpy as np
import datetime


def gen_id(prefix):
  return "{}_{}-{:03d}".format(
      prefix,
      datetime.datetime.now().isoformat(chr(ord("-"))).replace(
          ":", "-").replace(".", "-"), int(np.random.rand() * 1000))
