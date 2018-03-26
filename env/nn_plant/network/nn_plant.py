# -*- coding:utf-8 -*-
#  this structure is used to implement the nn plant structure.
#  nn plant is manual rnn
import numpy as np
import tensorflow as tf
import data.dataset as D
import network.core_nn as cn
import network.app_funcs as apf
from matplotlib import pyplot as plt


class Nplant:
    def __init__(self, config, val_dataX, val_dataY):
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
        #  read data in| process data:
        #  we only test one data
        self.datax, self.datay, self.init_state = D.r_sequence(val_dataX, val_dataY, self.config['m'])

    def step(self, input_state):
        input_state = input_state.reshape([1, -1])
        return self.sess.run(self.Y, feed_dict={self.X: input_state})

    def restore(self):
        #  I only need base structure.
        if self.config['first_rnn']:

            self.saver.restore(self.sess, self.config['pre_log_dir']['base'])
        else:
            self.saver.restore(self.sess, self.config['log_dir']['base'])

    def run_episode(self):
        #  run a whole episode:
        #  init_state is (x-(m-1)..., y-n)
        Ys = [self.init_state[-1]]
        input_data = np.concatenate((self.datax[1], self.init_state))
        for i in range(1, self.datax.shape[0]):
            input_data[:self.config['m']] = self.datax[i] 

            yn = self.step(input_data)
            yp = input_data[-1] 
            y = yp + yn
            input_data[self.config['m']:-1] = input_data[self.config['m']+1:]
            input_data[-1] = y

            Ys.append(y)

        return np.squeeze(np.array(Ys))

    def test(self):
        length = self.config['time_step']
        #  test a long episode run:
        Yp = self.run_episode()
        plt.plot(Yp[:length])
        plt.plot(self.datay[1:1+length])
        plt.show()

