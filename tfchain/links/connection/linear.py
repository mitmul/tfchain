from chainer import cuda

import tensorflow as tf
import tfchain


class Linear(tfchain.Link):

    def __init__(self, link):
        super(Linear, self).__init__(link)

    def forward(self, x):
        W = tf.transpose(self.params['W'])
        return tf.matmul(x, W) + self.params['b']
