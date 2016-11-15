import tensorflow as tf
import tfchain


class MaxPooling2D(tfchain.Function):

    def __init__(self, ksize, stride, pad):
        if isinstance(ksize, int):
            self.ksize = [1, ksize, ksize, 1]
        elif isinstance(ksize, tuple):
            self.ksize = [1] + list(ksize) + [1]
        else:
            raise AttributeError('Unknown data type of ksize')

        if stride is None:
            stride = ksize
        if isinstance(stride, int):
            self.stride = [1, stride, stride, 1]
        elif isinstance(stride, tuple) and len(stride) == 2:
            self.stride = [1] + list(stride) + [1]
        else:
            raise AttributeError('Unknown data type of stride')

        assert isinstance(pad, tuple) or isinstance(pad, int)
        if pad == (0, 0) or pad == 0:
            self.pad = 'VALID'
        else:
            self.pad = 'SAME'

    def forward(self, x):
        with tf.name_scope('MaxPooling2D', values=[x]):
            return tf.nn.max_pool(x, self.ksize, self.stride, self.pad)


def max_pooling_2d(x, ksize, stride=None, pad=0):
    return MaxPooling2D(ksize, stride, pad)(x)
