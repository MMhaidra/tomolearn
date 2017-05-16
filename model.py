import tensorflow as tf
from tf_wrappers import *


class Model(object):

    def __init__(self):
        self.layer_ls = []

    def add_conv_2d(self, n_filters, filter_size, use_pooling=True):
        self.layer_ls.append({'name': 'conv_2d',
                              'n_filters': n_filters,
                              'filter_size': filter_size,
                              'use_pooling': use_pooling})

    def add_fully_connected(self, n_out, use_relu=True):
        self.layer_ls.append({'name': 'fc',
                              'n_out': n_out,
                              'use_relu': use_relu})

    def compile_model(self, input):

        temp = input
        for ly in self.layer_ls:
            if ly['name'] == 'conv_2d':
                shape = (ly['filter_size'], ly['filter_size'], temp.shape[-1], ly['n_filters'])
                weights = new_weights(shape)
                biases = new_biases(ly['n_filters'])
                temp = tf.nn.conv2d(temp, weights, (1, 1, 1, 1), padding='SAME')
                temp += biases
                if ly['use_pooling']:
                    temp = tf.nn.max_pool(temp, (1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
                temp = tf.nn.relu(temp)
            elif ly['name'] == 'fc':
                try:
                    temp = flatten_layer(temp)
                except:
                    pass
                weights = new_weights([temp.shape[1], ly['n_out']])
                biases = new_biases(ly['n_out'])
                temp = tf.matmul(temp, weights) + biases
                if ly['use_relu']:
                    temp = tf.nn.relu(temp)
        return temp
