import tensorflow as tf


def on_cpu(f):
    def _f(*args, **kwargs):
        with tf.device('CPU'):
            return f(*args, **kwargs)
    return _f


def with_output(name, dtype=None, shape=None):
    def _with_output(f):
        f.output_types = getattr(f, 'output_types', {})
        f.output_types[name] = dtype

        f.output_shapes = getattr(f, 'output_shapes', {})
        f.output_shapes[name] = shape

        return f
    return _with_output
