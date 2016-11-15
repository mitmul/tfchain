from chainer import cuda

import chainer
import cupy
import numpy as np
import tensorflow as tf


class Function(object):

    def __call__(self, x):
        if isinstance(x, chainer.Variable):
            x = x.data
        if isinstance(x, cupy.ndarray):
            with cuda.Device(x.device):
                x = cuda.to_cpu(x)
        if x.ndim == 4:
            x = x.transpose(0, 2, 3, 1)  # to NHWC
        if isinstance(x, np.ndarray):
            x = tf.Variable(x)
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError
