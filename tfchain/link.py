from chainer import cuda

import chainer
import cupy
import numpy as np
import os
import tensorflow as tf


class Link(object):

    def __init__(self, link):
        self.params = dict([(os.path.basename(p[0]), p[1].data)
                            for p in link.namedparams()])

        for name, param in self.params.items():
            # Move to CPU
            if isinstance(param, cupy.ndarray):
                with cuda.Device(param.device):
                    self.params[name] = cuda.to_cpu(param)
            # Convert to tf.Variable
            self.params[name] = tf.Variable(self.params[name])

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
