"""
Proximal Policy Optimization (PPO) Class
zhuoxu@berkeley.edu
"""

import tensorflow as tf
import numpy as np


class PPO(object):

    def __init__(self, config):
        self.feed_dict = dict()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        g = tf.get_default_graph()
        self.sess = tf.Session(config=tf_config)
        self.config = config

        #  state:
        self.tfs = tf.placeholder(tf.float32, [None, self.config['s_dim']], 'state')
        self.feed_dict['state'] = self.tfs
    
        # critic:
        # input state, output normalized value
        with tf.variable_scope('critic'):
            self.v = self._build_cnet()
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'normalized_q_n')
            self.feed_dict['q_val'] = self.tfdc_r
            self.closs = tf.nn.l2_loss(self.tfdc_r - self.v)
            tf.summary.scalar('closs', self.closs)
            self.ctrain_op = tf.train.AdamOptimizer(self.config['c_lr']).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)       # choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, self.config['a_dim']], 'action')
        self.feed_dict['old_act'] = self.tfa
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        self.feed_dict['advantage'] = self.tfadv

        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                prob_new = tf.exp(tf.reduce_sum(pi.log_prob(self.tfa), axis=1))
                prob_old = tf.exp(tf.reduce_sum(oldpi.log_prob(self.tfa), axis=1))
                ratio = prob_new / prob_old
                surr = tf.multiply(ratio[:, np.newaxis], self.tfadv)
            self.aloss = -tf.reduce_sum(tf.minimum(
                surr,
                tf.clip_by_value(ratio[:, np.newaxis],
                                 1.-self.config['epsilon'],
                                 1.+self.config['epsilon'])*self.tfadv))

            tf.summary.scalar('aloss', self.aloss)

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(self.config['a_lr']).minimize(self.aloss)

        #  histogram of summary:
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            tf.summary.histogram(var.name, var)

        #  saver and summary:
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.trainable_variables())

        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.config['log_dir'], self.sess.graph)

    def get_feed_dict(self, results):
        feed_dict = {}
        for key, value in self.feed_dict.items():
            feed_dict[value] = results[key]
        return feed_dict

    def update(self, results):
        self.sess.run(self.update_oldpi_op)
        #  feed_dict:
        # update actor
        feed_dict = self.get_feed_dict(results) 
        [self.sess.run(self.atrain_op, feed_dict) 
                for _ in range(self.config['a_update_steps'])]
        # update critic
        [self.sess.run(self.ctrain_op, feed_dict)
                for _ in range(self.config['c_update_steps'])]

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            dim1 = 10 * self.config['s_dim']
            dim3 = 10 * self.config['a_dim']
            dim2 = int(np.sqrt(dim1 * dim3))
            l1 = tf.layers.dense(self.tfs, dim1, tf.nn.relu, trainable=trainable)
            l2 = tf.layers.dense(l1, dim2, tf.nn.relu, trainable=trainable)
            l3 = tf.layers.dense(l2, dim3, tf.nn.relu, trainable=trainable)
            mu = tf.layers.dense(l3, self.config['a_dim'], tf.nn.tanh, trainable=trainable,
                                 kernel_initializer=tf.random_normal_initializer(stddev=1/dim3))
            # log_sigma ~ N(-2,1/4), sigma has initial value of mean approximately exp(-2)
            log_sigma = tf.get_variable("log_sigma", shape=[self.config['a_dim']],
                                        initializer=tf.random_normal_initializer(mean=-2,
                                                                                 stddev=1/self.config['a_dim']))
            sigma = tf.exp(log_sigma)
            norm_dist = tf.contrib.distributions.Normal(mu, sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def _build_cnet(self):
        # neural net with three hidden layers
        dim1 = 10 * self.config['s_dim']
        dim3 = 10 * 1
        dim2 = int(np.sqrt(dim1 * dim3))
        l1 = tf.layers.dense(self.tfs, dim1, tf.nn.relu)
        l2 = tf.layers.dense(l1, dim2, tf.nn.relu)
        l3 = tf.layers.dense(l2, dim3, tf.nn.relu)
        return tf.layers.dense(l3, 1, kernel_initializer=tf.random_normal_initializer(stddev=1/dim3))

    def choose_action(self, s):
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -1, 1)

    def get_v(self, s):
        return self.sess.run(self.v, {self.tfs: s})

    def get_closs(self, s, r):
        return self.sess.run(self.closs, {self.tfs: s, self.tfdc_r: r})

    def get_aloss(self, s, a, adv):
        return self.sess.run(self.aloss, {self.tfs: s, self.tfa: a, self.tfadv: adv})

    def save(self):
        self.saver.save(self.sess, self.config['save_dir'])

    def restore(self):
        self.save.restore(self.sess, self.config['save_dir'] + '/latest')

    def log(self, results):
        feed_dict = self.get_feed_dict(results)
        summary_str = self.sess.run(self.summary_op, feed_dict)
        self.summary_writer.add_summary(summary_str)
