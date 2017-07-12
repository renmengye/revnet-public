from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np

from resnet.utils import logger

log = logger.get()


def cosine_angle(v1, v2):
  eps = 1e-7
  v1_norm = np.sqrt(np.dot(v1, v1))
  v2_norm = np.sqrt(np.dot(v2, v2))
  if v1_norm == 0.0 and v2_norm == 0.0:
    return 1.0 - eps, 1.0, 1.0
  if v1_norm == 0.0:
    v1_norm = 1.0
  if v2_norm == 0.0:
    v2_norm = 1.0
  cosine = np.dot(v1, v2) / v1_norm / v2_norm
  cosine = min(max(cosine, -1.0 + eps), 1.0 - eps)
  return cosine, v1_norm, v2_norm


def get_degree(radian):
  return radian * 180 / np.pi


def check_two_dict(d1, d2, tol=5e-1, name=None):
  """Check two dictionaries."""
  assert len(d1) == len(d2), "Dictionary length not equal."
  if name is not None:
    log.info(name)
  log.info("{:35s} {:10s} {:10s} {:10s}".format("Name", "Norm", "Cosine",
                                                "Angle"))
  log.info("-" * 80)
  keys = sorted(d1.keys())
  for kk in keys:
    x1 = d1[kk].ravel()
    x2 = d2[kk].ravel()
    cosine, v1_norm, v2_norm = cosine_angle(x1, x2)
    norm = v1_norm / v2_norm
    angle = get_degree(np.arccos(cosine))
    if angle < tol:
      log.info("{:35s} {:10.4e} {:10.4e} {:10.4e}".format(kk, norm, cosine,
                                                          angle))
    else:
      log.error("{:35s} {:10.4e} {:10.4e} \033[31m{:10.4e}\033[39m".format(
          kk, norm, cosine, angle))
  d1_all = np.concatenate([d1[kk].ravel() for kk in keys])
  d2_all = np.concatenate([d2[kk].ravel() for kk in keys])
  cosine, v1_norm, v2_norm = cosine_angle(d1_all, d2_all)
  norm = v1_norm / v2_norm
  angle = get_degree(np.arccos(cosine))
  log.info("-" * 80)
  if angle < tol:
    log.info("{:35s} {:10.4e} {:10.4e} {:10.4e}".format("total", norm, cosine,
                                                        angle))
  else:
    log.error("{:35s} {:10.4e} {:10.4e} \033[31m{:10.4e}\033[39m".format(
        "total", norm, cosine, angle))
