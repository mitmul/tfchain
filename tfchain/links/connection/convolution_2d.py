import chainer
import numpy as np
import tensorflow as tf
import tfchain


class Convolution2D(tfchain.Link):

    def __init__(self, *args):
        if len(args) == 1:
            super(Convolution2D, self).__init__(*args)
        if len(args) == 4:
            super(Convolution2D, self).__init__(*args[:2])
        self.W = tf.transpose(self.W, perm=[2, 3, 1, 0])
        self.b = self.b[np.newaxis, np.newaxis, np.newaxis, :]

        def convert_stride(stride):
            if isinstance(stride, int):
                self.stride = [1, stride, stride, 1]
            elif isinstance(stride, tuple) and len(stride) == 2:
                self.stride = [1] + list(stride) + [1]
            else:
                raise AttributeError('Unknown data type of args.stride')

        def convert_pad(pad):
            assert isinstance(pad, tuple) or isinstance(pad, int)
            if pad == (0, 0) or pad == 0:
                self.pad = 'VALID'
            else:
                self.pad = 'SAME'

        if len(args) == 1 and isinstance(args[0], chainer.Link):
            convert_stride(args[0].stride)
            convert_pad(args[0].pad)
        elif len(args) == 4:
            convert_stride(args[2])
            convert_pad(args[3])

    def forward(self, x):
        return tf.nn.conv2d(x, self.W, self.stride, self.pad) + self.b
