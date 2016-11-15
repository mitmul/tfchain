#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tfchain.session import get_session

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import tensorflow as tf
import tfchain


class LeNet5(chainer.Chain):

    def __init__(self):
        super(LeNet5, self).__init__(
            conv1=L.Convolution2D(1, 6, 5),
            conv2=L.Convolution2D(5, 16, 5),
            fc3=L.Linear(None, 120),
            fc4=L.Linear(120, 84),
            fc5=L.Linear(84, 10)
        )
        self.train = True

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2, 2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2, 2)
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc4(h))
        h = F.relu(self.fc5(h))
        return h


if __name__ == '__main__':
    model = LeNet5()
