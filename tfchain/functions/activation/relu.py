import tensorflow as tf
import tfchain


class ReLU(tfchain.Function):

    def forward(self, x):
        return tf.nn.relu(x)


def relu(x):
    return ReLU()(x)
