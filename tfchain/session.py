import os
import tensorflow as tf

_SESSION = None


def get_session():
    global _SESSION
    session = tf.get_default_session()
    if session is None:
        if _SESSION is None:
            n_threads = os.environ.get('OMP_NUM_THREADS')
            if n_threads is None:
                config = tf.ConfigProto(allow_soft_placement=True)
            else:
                config = tf.ConfigProto(intra_op_parallelism_threads=n_threads,
                                        allow_soft_placement=True)
            _SESSION = tf.Session(config=config)
        session = _SESSION
    else:
        session = tf.get_default_session()
    return session
