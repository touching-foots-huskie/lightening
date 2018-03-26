# : Harvey Chang
# : chnme40cs@gmail.com
#  Gru rnn is rnn structure for error structure
import tensorflow as tf
import numpy as np


class Gru_rnn(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_units, state_dim, reuse=None):
        super(Gru_rnn, self).__init__(_reuse=reuse)
        self._num_units = num_units  # num units is the dimension
        self.state_dim = state_dim

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, input, state):
        #  input and state are x and h:
        #  x and h are all m dimension
        x = tf.layers.dense(input, self.state_dim,  activation=tf.nn.leaky_relu, kernel_initializer=tf.contrib.keras.initializers.glorot_uniform()) 
        h = tf.layers.dense(state, self.state_dim,  activation=tf.nn.leaky_relu, kernel_initializer=tf.contrib.keras.initializers.glorot_uniform()) 
        #  r gate:
        xr = tf.layers.dense(input, self.state_dim, activation=tf.nn.leaky_relu, kernel_initializer=tf.contrib.keras.initializers.glorot_uniform()) 
        hr = tf.layers.dense(state, self.state_dim, activation=tf.nn.leaky_relu, kernel_initializer=tf.contrib.keras.initializers.glorot_uniform()) 
        r = tf.nn.sigmoid(xr + hr)
        #  z gate:
        xz = tf.layers.dense(input, self.state_dim, activation=tf.nn.leaky_relu, kernel_initializer=tf.contrib.keras.initializers.glorot_uniform()) 
        hz = tf.layers.dense(state, self.state_dim, activation=tf.nn.leaky_relu, kernel_initializer=tf.contrib.keras.initializers.glorot_uniform()) 
        z = tf.nn.sigmoid(xz + hz)
        #  update gate: merge:
        h = tf.nn.tanh(x + r*h)
        h = (1-z)*h + z*state
        #  h means to be (y3, y2, y1. etc)
        return h[:, -1:], h
