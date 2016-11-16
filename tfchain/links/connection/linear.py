import chainer
import numpy as np
import tensorflow as tf
import tfchain


class Linear(tfchain.Link):

    def __init__(self, W, b):
        super(Linear, self).__init__(W, b)

    def forward(self, x):
        if isinstance(x, tf.Tensor):
            shape = x.get_shape()
            if len(shape) != 2:
                x = tf.reshape(x, (int(shape[0]), int(np.prod(shape[1:]))))
        elif isinstance(x, chainer.Variable):
            shape = x.shape
            x = x.reshape((shape[0], np.prod(shape[1:])))
        with tf.name_scope('Linear', values=[x, self.W, self.b]):
            return tf.matmul(x, self.W) + self.b
