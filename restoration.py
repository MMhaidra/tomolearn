import os

import tensorflow as tf
import numpy as np
import dxchange

from util import *
from model import Model


class Image_Restoration(object):

    def __init__(self):

        self.model = None
        self.x_train_path = None
        self.y_ref_path = None
        self.sess = tf.Session()

    def load_model(self, model):

        assert isinstance(model, Model)
        self.model = model

    def train(self, x_train_path, y_ref_path, coeff_tv=0.3, n_iter=100, batch_szie=50, save_path='checkpoints/',
              save_interval=50):

        self.x_train_path = x_train_path
        self.y_ref_path = y_ref_path
        temp = load_tiff_stack(x_train_path, rand_batch_size=1)
        self.image_shape = temp.shape[1:]
        image_size_flat = np.prod(self.image_shape)

        x_train = tf.placeholder(tf.float32, (None, image_size_flat), name='x_train')
        x_train_image = tf.reshape(x_train, (None, self.image_shape[0], self.image_shape[1], 1))
        y_ref = tf.placeholder(tf.float32, (None, image_size_flat), name='y_ref')
        y_ref_image = tf.reshape(y_ref, (None, self.image_shape[0], self.image_shape[1], 1))

        y_train = self.model.compile_model(x_train_image)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        cost = tf.contrib.losses.sum_of_squares(y_train, y_ref) + coeff_tv * tf.image.total_variation(y_train)
        optimizer = tf.train.AdamOptimizer().minimize(cost, global_step=global_step)

        saver = tf.train.Saver()
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        ifrerun = True
        last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_path)
        if last_chk_path is not None:
            print('Restore last checkpoint? (y/n)')
            ifload = raw_input()
            if ifload == 'y' or 'Y':
                try:
                    saver.restore(self.sess, last_chk_path)
                    ifrerun = False
                except:
                    print('An error occurred. Starting from scratch instead...')
        if ifrerun:
            self.sess.run(tf.global_variables_initializer())

        for i in range(n_iter):

            # get training data
            x_batch, y_ref_batch = load_tiff_data(x_train_path, y_ref_path, rand_batch_size=50)
            x_batch = tf.convert_to_tensor(x_batch[:, :, :, np.newaxis])
            y_ref_batch = tf.convert_to_tensor(y_ref_batch[:, :, :, np.newaxis])

            feed_dict_train = {x_train_image: x_batch, y_ref_image: y_ref_batch}

            i_global, _ = self.sess.run([global_step, optimizer], feed_dict=feed_dict_train)
            print('Current step (global): {:d}'.format(i_global))

            if i_global % save_interval == 0 or i_global == n_iter - 1:
                saver.save(self.sess, save_path, global_step=global_step)

            if i_global == n_iter - 1:
                break


if __name__ == '__main__':

    model = Model()
    model.add_conv_2d(64, 9)
    model.add_conv_2d(32, 9)
    model.add_conv_2d(1, 9, use_pooling=False)
    processor = Image_Restoration()
    processor.load_model(model)
    processor.train('data/train/x', 'data/train/y', coeff_tv=0.3, n_iter=100, batch_szie=20)
