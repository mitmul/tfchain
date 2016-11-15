import chainer
import numpy as np
import tensorflow as tf
import tfchain


class Convolution2D(tfchain.Link):

    def __init__(self, W, b, stride, pad):
        super(Convolution2D, self).__init__(W, b)

        if isinstance(stride, int):
            self.stride = [1, stride, stride, 1]
        elif isinstance(stride, tuple) and len(stride) == 2:
            self.stride = [1] + list(stride) + [1]
        else:
            raise AttributeError('Unknown data type of args.stride')

        assert isinstance(pad, tuple) or isinstance(pad, int)
        if pad == (0, 0) or pad == 0:
            self.pad = 'VALID'
        else:
            self.pad = 'SAME'

    def forward(self, x):
        with tf.name_scope('Convolution2D', values=[x, self.W, self.b]):
            return tf.nn.conv2d(x, self.W, self.stride, self.pad) + self.b
