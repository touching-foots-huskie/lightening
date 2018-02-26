# -*- coding:utf-8 -*-
#  this file is going to construct a whole network:
import numpy as np
import tensorflow as tf
import network.core_nn as cn
import network.app_funcs as apf
import network.deep_rnn as drnn

class Pdn:
    def __init__(self, config):
        self.config = config
        self.typ = self.config['typ']
        
        #  dimension settings:
        self.batch_size = config['batch_size']
        self.time_step = config['time_step']
        self.channel = config['channel']

        #  core setting:
        self.core_nn = cn.nn_wrapper(config['core_nn'])
        self.app_func = apf.app_wrapper(config['app_func'])

        self.data_dict = dict()

        if self.typ == 'pre':
            with tf.variable_scope('rnn/deep_rnn'):
                self.output = self.pre_version()
        elif self.typ == 'rnn':
            self.output = self.rnn_version()

        self.data_dict['X'] = self.X
        self.data_dict['Y'] = self.Y
        #  add append structure 
        if self.config['append']:
            mid_output = self.append_version()
            self.output += mid_output
            self.data_dict['Xp'] = self.Xp

        #  loss structure
        if self.config['target'] == 'action':
            self.loss = tf.nn.l2_loss((self.Y - self.output))
        elif self.config['target'] == 'classification':
            self.loss = tf.nn.softmax_cross_entropy_with_logits(self.output, self.Y)

        self.opt = tf.train.AdagradOptimizer(learning_rate=self.config['learning_rate'], initial_accumulator_value=1e-6)
        self.train_op = self.opt.minimize(self.loss)

        self.param_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(self.param_list)
        self.init = tf.global_variables_initializer()
        
        #  run structure:
        self.sess = tf.Session()
        self.sess.run(self.init)

    def pre_version(self):
        #  get the input tensor:
        self.X = tf.placeholder(tf.float32, [self.batch_size, self.time_step, self.channel])
        self.Y = tf.placeholder(tf.float32, [self.batch_size, self.time_step, 1])
        outputs = self.core_nn(self.X)
        return outputs

    def rnn_version(self):

        self.X = tf.placeholder(tf.float32, [self.batch_size, self.time_step, self.channel])
        self.Y = tf.placeholder(tf.float32, [self.batch_size, self.time_step, 1])

        cell = drnn.Deep_rnn(1, self.core_nn)
        init_state = tf.constant(np.zeros([self.batch_size, self.config['n']]), dtype=tf.float32)  #  n is y dimension
        outputs, final_state = tf.nn.dynamic_rnn(cell, self.X, initial_state=init_state,
                                                 time_major=False, swap_memory=True)
        return outputs

    def append_version(self):
        #  this part is the append part:
        self.Xp = tf.placeholder(tf.float32, [self.batch_size, self.time_step, self.config['p_channel']])
        output = self.app_func(self.Xp)
        return output

    def get_feed_dict(self, data):
        feed_dict = dict()
        for key, value in data.items():
            if key in self.data_dict.keys():
                feed_dict[self.data_dict[key]] = value
        return feed_dict

    def update(self, data):
        feed_dict = self.get_feed_dict(data)
        return self.sess.run([self.train_op, self.loss, self.output], feed_dict)

    def validate(self, data):
        feed_dict = self.get_feed_dict(data)
        return self.sess.run([self.loss, self.output], feed_dict)

    def restore(self):
        if (self.config['typ'] == 'rnn' and self.config['first_rnn']):
            
            self.saver.restore(self.sess, self.config['pre_log_dir'])
        else:
            self.saver.restore(self.sess, self.config['log_dir'])
            print('model loaded!')
           
    def save(self):
        self.saver.save(self.sess, self.config['log_dir'])
        print('model saved')

