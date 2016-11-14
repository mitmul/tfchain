#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages
from setuptools import setup

setup_requires = []
install_requires = [
    'chainer>=1.17.0',
    'tensorflow>=0.11.0rc2',
    'numpy>=1.9.0',
    'protobuf',
    'six>=1.9.0',
]

setup(name='tfchain',
      version='0.0.1',
      description='Use TensorFlow as a backend of a Chainer\'s chain',
      author='Shunta Saito',
      author_email='shunta@preferred.jp',
      packages=['tfchain'],
      install_requires=install_requires,
      )
