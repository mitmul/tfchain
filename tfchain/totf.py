import heapq
from operator import itemgetter as ig

import chainer
import numpy as np
import tensorflow as tf
import tfchain.functions as F
import tfchain.links as L
from tfchain import session


def totf(forward):

    def f(model, x):
        feed_x = x.data
        if x.ndim == 4:
            feed_x = feed_x.transpose(0, 2, 3, 1)
            shape = feed_x.shape
        else:
            shape = x.shape

        if not hasattr(model, 'tf_graph'):
            model.input_x = tf.placeholder(x.dtype, shape)
            y = forward(model, x)
            cand_funcs = []
            comp_graph = []
            seen_set = set()

            def add_cand(cand):
                if cand not in seen_set:
                    # Negate since heapq is min-heap
                    heapq.heappush(
                        cand_funcs, (-cand.rank, len(seen_set), cand))
                    seen_set.add(cand)

            add_cand(y.creator)
            while cand_funcs:
                _, _, func = heapq.heappop(cand_funcs)
                comp_graph.append((func, func.inputs[1:]))
                for func_x in func.inputs:
                    if func_x.creator is not None:
                        add_cand(func_x.creator)

            model.tf_graph = []
            for link, param in reversed(comp_graph):
                label = link.label
                if label == 'Convolution2DFunction':
                    param = param + [(link.sy, link.sx)] + [(link.ph, link.pw)]
                    model.tf_graph.append(L.Convolution2D(*param))
                elif label == 'LinearFunction':
                    model.tf_graph.append(L.Linear(*param))
                elif label == 'MaxPooling2D':
                    model.tf_graph.append(F.MaxPooling2D(
                        (link.kh, link.kw), (link.sy, link.sx),
                        (link.ph, link.pw)))
                elif label == 'ReLU':
                    model.tf_graph.append(F.ReLU())

            y = model.input_x
            for f in model.tf_graph:
                y = f(y)
            model.op = y

        if not hasattr(model, 'session'):
            model.session = session.get_session()
            model.session.run(tf.initialize_all_variables())

        return model.session.run(model.op, feed_dict={model.input_x: feed_x})

    return f
