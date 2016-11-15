import os

import chainer
import chainer.functions as F
import cupy
import numpy as np
import tensorflow as tf
from chainer import cuda


class Link(object):

    def __init__(self, W, b):
        assert isinstance(W, chainer.Variable)
        assert isinstance(b, chainer.Variable)

        if W.ndim == 4:
            # When it's convolution kernel
            W = F.transpose(W, axes=(2, 3, 1, 0))
        elif W.ndim == 2:
            # When it's affine matrix
            W = F.transpose(W)
        else:
            raise TypeError('Unsupported shape of input W: {}'.format(W.shape))

        setattr(self, 'W', tf.Variable(W.data, name='W'))
        setattr(self, 'b', tf.Variable(b.data, name='b'))

    def __call__(self, x):
        if isinstance(x, chainer.Variable):
            x = x.data
        if isinstance(x, cupy.ndarray):
            with cuda.Device(x.device):
                x = cuda.to_cpu(x)
        if hasattr(x, 'ndim') and x.ndim == 4:
            x = x.transpose(0, 2, 3, 1)  # to NHWC
        if isinstance(x, np.ndarray):
            x = tf.Variable(x)
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError
