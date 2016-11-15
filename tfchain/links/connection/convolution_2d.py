import numpy as np
import tensorflow as tf
import tfchain


class Convolution2D(tfchain.Link):

    def __init__(self, link):
        super(Convolution2D, self).__init__(link)
        self.params['W'] = tf.transpose(self.params['W'], perm=[2, 3, 1, 0])
        self.params['b'] = self.params['b'][
            np.newaxis, np.newaxis, np.newaxis, :]

        assert isinstance(link.pad, tuple) or isinstance(link.pad, int)
        if link.pad == (0, 0) or link.pad == 0:
            self.pad = 'VALID'
        else:
            self.pad = 'SAME'

        if isinstance(link.stride, int):
            self.stride = [1, link.stride, link.stride, 1]
        elif isinstance(link.stride, tuple) and len(link.stride) == 2:
            self.stride = [1] + list(link.stride) + [1]
        else:
            raise AttributeError('Unknown data type of link.stride')

    def forward(self, x):
        h = tf.nn.conv2d(x, self.params['W'], self.stride, self.pad)
        h = h + self.params['b']
        return h
