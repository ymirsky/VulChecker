#!/usr/bin/env python

from Cython.Build import cythonize
from setuptools import Extension, setup

setup(
    ext_modules=cythonize(
        [
            Extension("hector_ml._features", ["src/hector_ml/_features.pyx"]),
            Extension("hector_ml._graphs", ["src/hector_ml/_graphs.pyx"]),
        ]
    )
)
