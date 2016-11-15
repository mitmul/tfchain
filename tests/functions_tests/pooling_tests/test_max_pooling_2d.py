#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import testing
from tfchain.session import get_session

import chainer.functions as cF
import numpy as np
import tensorflow as tf
import tfchain.functions as tF
import unittest


class TestMaxPooling2D(unittest.TestCase):

    def setUp(self):
        self.x = np.random.randn(1, 2, 4, 4).astype(np.float32)

    def test_forward(self):
        cy = cF.max_pooling_2d(self.x, 2, 2).data
        ty = tF.max_pooling_2d(self.x, 2, 2)
        sess = get_session()
        sess.run(tf.initialize_all_variables())
        ty = ty.eval(session=sess)
        ty = ty.transpose(0, 3, 1, 2)
        testing.assert_allclose(ty, cy)
