#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import testing
from tfchain.session import get_session

import chainer.links as cL
import numpy as np
import tensorflow as tf
import tfchain.links as tL
import unittest


class TestConvolution2D(unittest.TestCase):

    def setUp(self):
        self.x = np.random.rand(1, 1, 28, 28).astype(np.float32)
        self.cconv = cL.Convolution2D(1, 6, 5)

    def test_link_forward(self):
        tconv = tL.Convolution2D(self.cconv)
        ty = tconv(self.x)
        sess = get_session()
        sess.run(tf.initialize_all_variables())
        ty = ty.eval(session=sess)
        ty = ty.transpose(0, 3, 1, 2)  # to NCHW
        cy = self.cconv(self.x).data
        testing.assert_allclose(ty, cy)

    def test_param_forward(self):
        tconv = tL.Convolution2D(
            self.cconv.W, self.cconv.b, self.cconv.stride, self.cconv.pad)
        ty = tconv(self.x)
        sess = get_session()
        sess.run(tf.initialize_all_variables())
        ty = ty.eval(session=sess)
        ty = ty.transpose(0, 3, 1, 2)  # to NCHW
        cy = self.cconv(self.x).data
        testing.assert_allclose(ty, cy)
