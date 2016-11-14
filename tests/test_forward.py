#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import tfchain
import unittest


class TestForward(unittest.TestCase):

    def setUp(self):
        self.b = 1
        self.n = 5
        self.model = chainer.Chain(
            l1=L.Linear(self.n, self.n),
            l2=L.Linear(self.n, self.n)
        )
        self.x = np.random.rand(self.b, self.n).astype('f')

    def test_forward(self):
        h = self.model.l1(self.x)
        y = self.model.l2(h)
        assert y.shape == (self.b, self.n)

        # session = tfchain.session.get_session()
