from tfchain import session

import chainer
import heapq
import numpy as np
import tensorflow as tf
import tfchain.functions as F
import tfchain.links as L


def totf(forward):

    def f(model, x):
        if not hasattr(model, 'tf_graph'):
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
                for input_x in func.inputs:
                    if input_x.creator is not None:
                        add_cand(input_x.creator)

            model.tf_graph = []
            for link, param in reversed(comp_graph):
                label = link.label
                if label == 'Convolution2DFunction':
                    param = param + [(link.sy, link.sx)] + [(link.ph, link.pw)]
                    model.tf_graph.append(L.Convolution2D(*param))
                elif label == 'LinearFunction':
                    model.tf_graph.append(L.Linear(*param))
                elif label == 'MaxPooling2D':
                    ksize = (link.kh, link.kw)
                    stride = (link.sy, link.sx)
                    pad = (link.ph, link.pw)
                    model.tf_graph.append(F.MaxPooling2D(ksize, stride, pad))
                elif label == 'ReLU':
                    model.tf_graph.append(F.ReLU())

        for f in model.tf_graph:
            if isinstance(f, L.Linear):
                if isinstance(x, tf.Tensor):
                    shape = x.get_shape()
                    x = tf.reshape(x, (int(shape[0]), int(np.prod(shape[1:]))))
                elif isinstance(x, chainer.Variable):
                    shape = x.shape
                    x = x.reshape((shape[0], np.prod(shape[1:])))

            x = f(x)

        if not hasattr(model, 'session'):
            sess = session.get_session()
            sess.run(tf.initialize_all_variables())
            model.session = sess

        return model.session.run(x)

    return f
