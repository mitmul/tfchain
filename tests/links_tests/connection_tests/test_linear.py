#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import chainer.links as cL
import numpy as np
import tensorflow as tf
import tfchain.links as tL
from chainer import testing
from tfchain.session import get_session


class TestLinear(unittest.TestCase):

    def setUp(self):
        self.clinear = cL.Linear(100, 10)

    def test_param_forward(self):
        tlinear = tL.Linear(self.clinear.W, self.clinear.b)
        x = np.random.rand(1, 100).astype(np.float32)
        cy = self.clinear(x).data
        ty = tlinear(x)
        sess = get_session()
        sess.run(tf.initialize_all_variables())
        ty = ty.eval(session=sess)
        testing.assert_allclose(ty, cy)
