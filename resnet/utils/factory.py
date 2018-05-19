"""Factory implementation."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


class Factory(object):
  """Factory implementation."""

  def __init__(self):
    self._registry = dict()

  def register(self, name):
    """Register a class."""

    def decorator(f):
      self._registry[name] = f
      return f

    return decorator

  def has(self, name):
    """Check if a name has been registered."""
    return name in self._registry

  def create(self, name, *args, **kwargs):
    """Create a class."""
    return self._registry[name](*args, **kwargs)

  def get(self, name):
    """Get a class constructor."""
    return self._registry[name]
