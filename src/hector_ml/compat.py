"""Backfills for features added in newer versions of Python."""

__all__ = ("cached_property",)

# functools.cached_property was introduced in Python 3.8
try:
    from functools import cached_property
except ImportError:

    class cached_property:
        def __init__(self, fn):
            self.fn = fn

        def __get__(self, instance, owner=None):
            if instance is None:
                return self

            cache = instance.__dict__
            try:
                return cache[self.fn.__name__]
            except KeyError:
                val = cache[self.fn.__name__] = self.fn(instance)
                return val
