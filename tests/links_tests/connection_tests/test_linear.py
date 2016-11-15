#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import testing
from tfchain.session import get_session

import chainer.links as cL
import numpy as np
import tensorflow as tf
import tfchain.links as tL
import unittest


class TestLinear(unittest.TestCase):

    def setUp(self):
        self.clinear = cL.Linear(100, 10)
        self.tlinear = tL.Linear(self.clinear)

    def test_forward(self):
        x = np.random.rand(1, 100).astype(np.float32)
        cy = self.clinear(x).data
        ty = self.tlinear(x)
        sess = get_session()
        sess.run(tf.initialize_all_variables())
        ty = ty.eval(session=sess)
        testing.assert_allclose(ty, cy)
