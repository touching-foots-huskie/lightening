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

        self.param_dict = dict()
        self.data_dict = dict()
        self.saver_dict = dict()

        if self.typ == 'pre':
            with tf.variable_scope('rnn/deep_rnn'):
                self.output = self.pre_version()
        elif self.typ == 'rnn':
            self.output = self.rnn_version()

        #  base param_part
        self.param_dict['base'] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'rnn/deep_rnn')

        self.data_dict['X'] = self.X
        self.data_dict['Y'] = self.Y
        #  add append structure 
        if self.config['append']:
            with tf.variable_scope('append') as scope:
                mid_output = self.append_version()
                self.output += mid_output
                self.param_dict['append'] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'append')
            self.data_dict['Xp'] = self.Xp

        #  loss structure
        if self.config['target'] == 'action':
            self.loss = tf.nn.l2_loss((self.Y - self.output))
            self.pred = self.output

        elif self.config['target'] == 'classification':
            self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.label, logits=self.output)
            self.pred = tf.argmax(self.output, -1),
            correction = tf.equal(self.pred, tf.argmax(self.label, -1))
            self.accuracy = tf.reduce_mean(tf.cast(correction, tf.float32))

        self.opt = tf.train.AdagradOptimizer(learning_rate=self.config['learning_rate'], initial_accumulator_value=1e-6)

        if self.config['update_part'] == 'all':
            self.train_op = self.opt.minimize(self.loss)
        else:
            self.grads = tf.gradients(self.loss, self.param_dict[self.config['update_part']])
            self.train_op = self.opt.apply_gradients(zip(self.grads, self.param_dict[self.config['update_part']]))

        #  savers setting:
        for key, value in self.param_dict.items():
            self.saver_dict[key] = tf.train.Saver(value)

        self.init = tf.global_variables_initializer()
        
        #  run structure:
        self.sess = tf.Session()
        self.sess.run(self.init)

    def pre_version(self):
        #  get the input tensor:
        self.X = tf.placeholder(tf.float32, [self.batch_size, self.time_step, self.channel])

        if self.config['target'] == 'classification':
            self.Y = tf.placeholder(tf.int32, [self.batch_size, self.time_step])
            self.label = tf.one_hot(self.Y, self.config['classes'], axis=-1)
            outputs = self.core_nn(self.X, self.config['classes'])
        else:
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
        self.Xp = tf.placeholder(tf.float32, [self.batch_size, self.time_step, self.config['channel']])
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
        if self.config['target'] == 'classification':
            return self.sess.run([self.train_op, self.accuracy, self.pred], feed_dict)
        else:
            return self.sess.run([self.train_op, self.loss, self.pred], feed_dict)

    def validate(self, data):
        feed_dict = self.get_feed_dict(data)
        if self.config['target'] == 'classification':
            return self.sess.run([self.accuracy, self.pred], feed_dict)
        else:
            return self.sess.run([self.loss, self.pred], feed_dict)

    def restore(self, name='base'):
        if (self.config['typ'] == 'rnn' and self.config['first_rnn']):
            
            self.saver_dict[name].restore(self.sess, self.config['pre_log_dir'][name])
            
        else:
            self.saver_dict[name].restore(self.sess, self.config['log_dir'][name])
        print('{} model loaded!'.format(name))
           
    def save(self):
        for key, saver in self.saver_dict.items():
            saver.save(self.sess, self.config['log_dir'][key])
            #  save all the model
            print('{} model saved'.format(key))

