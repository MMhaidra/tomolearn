import tensorflow as tf


def new_weights(shape):

    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):

    return tf.Variable(tf.constant(0.05, shape=[length]))


def flatten_layer(layer):

    shape = layer.get_shape()
    n_features = shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, n_features])
    return layer_flat