from chainer import cuda

import chainer
import cupy
import numpy as np
import os
import tensorflow as tf


class Link(object):

    def __init__(self, *args):
        # When given a link
        if len(args) == 1 and isinstance(args[0], chainer.Link):
            params = dict([(os.path.basename(p[0]), p[1].data)
                           for p in args[0].namedparams()])
            for name, param in params.items():
                # Move to CPU
                if isinstance(param, cupy.ndarray):
                    with cuda.Device(param.device):
                        params[name] = cuda.to_cpu(param)
                # Convert to tf.Variable
                setattr(self, name, tf.Variable(params[name]))

        # When given W, b explicitly as Variables
        elif len(args) == 2 and isinstance(args[0], chainer.Variable) \
                and isinstance(args[1], chainer.Variable):
            setattr(self, 'W', tf.Variable(args[0].data))
            setattr(self, 'b', tf.Variable(args[1].data))

        else:
            raise TypeError('Wrong number of arguments. {}'.format(args))

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
