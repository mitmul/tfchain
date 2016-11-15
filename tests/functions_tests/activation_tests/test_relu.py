#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import testing
from tfchain.session import get_session

import chainer.functions as cF
import numpy as np
import tensorflow as tf
import tfchain.functions as tF
import unittest


class TestReLU(unittest.TestCase):

    def setUp(self):
        self.x = np.random.randn(1, 10).astype(np.float32)

    def test_forward(self):
        cy = cF.relu(self.x).data
        ty = tF.relu(self.x)
        sess = get_session()
        sess.run(tf.initialize_all_variables())
        ty = ty.eval(session=sess)
        testing.assert_allclose(ty, cy)
