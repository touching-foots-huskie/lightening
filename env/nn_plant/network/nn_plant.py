# -*- coding:utf-8 -*-
#  this structure is used to implement the nn plant structure.
import numpy as np
import tensorflow as tf
import network.core_nn as cn
import network.app_funcs as apf
from matplotlib import pyplot as plt


class Nplant:
    def __init__(self, config):
        self.config = config
        #  core funcs:
        self.core_nn = cn.nn_wrapper(config['core_nn']) 
        with tf.variable_scope('rnn/deep_rnn') as scope:
            self.X = tf.placeholder(tf.float32, [1, self.config['channel']])  
            #  networks:
            self.Y = self.core_nn(self.X)

        #  saver structures:
        self.param_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'rnn/deep_rnn')
        self.saver = tf.train.Saver(self.param_list) 

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def step(self, input_state):
        input_state = input_state.reshape([1, -1])
        return self.sess.run(self.Y, feed_dict={self.X: input_state})/1e5

    def restore(self):
        #  I only need base structure.
        self.saver.restore(self.sess, self.config['log_dir']['base'])

    def process_data(self, X, Y, x):
        #  X (m,)
        #  Y (n,)
        X = np.concatenate((X[1:], np.array(x).reshape(1)))
        differ_x = X[1:] - X[:-1]
        differ_y = Y[1:] - Y[:-1]
        input_state = np.concatenate((1e2*differ_x, np.array(x).reshape(1), 1e4*differ_y, Y[-1:]), axis=0)
        return X, input_state 

    def run_episode(self, Xs, init_state):
        #  run a whole episode:
        #  init_state is (x-(m-1)..., y-n)
        X = init_state[:self.config['m']]
        Y = init_state[self.config['m']:]
        Ys = []
        for x in Xs:
            X, input_data = self.process_data(X, Y, x)
            yn = self.step(input_data)
            yp = Y[-1]*2 - Y[-2]
            y = yp + yn

            Y = np.concatenate((Y[1:], np.array(y).reshape(1)))
            Ys.append(y)

        return np.squeeze(np.array(Ys))

    def test(self, X, Y, length=10):
        #  test a long episode run:
        init_state = np.concatenate((X[10-self.config['m']:10], Y[10-self.config['n']:10]))

        Yp = self.run_episode(X[10:], init_state)
        plt.plot(Yp[:length])
        plt.plot(Y[10:10+length])
        plt.show()

