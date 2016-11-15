import tensorflow as tf
import tfchain


class Linear(tfchain.Link):

    def __init__(self, *args):
        super(Linear, self).__init__(*args)

    def forward(self, x):
        print(self.W.get_shape())
        print(self.b.get_shape())
        return tf.matmul(x, tf.transpose(self.W)) + self.b
