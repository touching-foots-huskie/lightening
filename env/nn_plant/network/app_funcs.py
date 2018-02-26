# : Harvey Chang
# : chnme40cs@gmail.com
# app_funcs focused on many appending functions:

import tensorflow as tf
import numpy as np


def app_wrapper(No):
    return {
        1:ln1,
        2:ln2
    }[No]


# Linear versions:
def ln1(X):
    W_value = [0.6, 0.3]
    channel = len(W_value)
    W = tf.constant(np.array(W_value).reshape([1, 1, channel]))
    W = tf.cast(W, tf.float32)
    # repeat:
    W = tf.tile(W, [X.shape[0].value, X.shape[1].value, 1])
    output = tf.reduce_sum(W*X, axis=-1)
    # reshape
    output = tf.expand_dims(output, -1)
    return output


def ln2(layer):
    hidden_dim = [32, 64, 64, 32]

    for dim in hidden_dim:
        layer = tf.layers.dense(layer, dim,
                                activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.keras.initializers.glorot_uniform())
    output = tf.layers.dense(layer, 1, kernel_initializer=tf.contrib.keras.initializers.glorot_uniform())

    return output


if __name__ == '__main__':
    input = tf.ones([10, 100, 3])
    output = ln1(input)
    print(output.shape)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(output))
