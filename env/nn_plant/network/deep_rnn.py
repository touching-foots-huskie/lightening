# : Harvey Chang
# : chnme40cs@gmail.com
# this file is used to design a specific rnn:
import tensorflow as tf
import numpy as np


class Deep_rnn(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_units, core_nn, reuse=None):
        super(Deep_rnn, self).__init__(_reuse=reuse)
        self._num_units = num_units  # num units is the dimension
        self._core_nn = core_nn

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, input, state):
        #  input and state are in same shape
        #  state are the ys.
        _input = tf.concat([input, state], axis=-1)
        _error = self._core_nn(_input)
        previous_y = state[:, 1:]  # throw the old ones
        next_y = state[:, -1:] + _error

        new_state = tf.concat([previous_y, next_y], axis=-1)  # add the new ones
        return next_y, new_state
