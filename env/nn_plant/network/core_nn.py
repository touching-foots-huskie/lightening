# : Harvey Chang
# : chnme40cs@gmail.com
# this core_nn has most of the widely used nn structure:
import tensorflow as tf
import numpy as np


def nn_wrapper(No):
    return {1:nn1, 2:nn2, 3:nn3, 4:nn4, 5:nn5}[No]


def nn1(layer):
    hidden_dim = [32, 64, 64, 32]
    #  add bn:
    for dim in hidden_dim:
        layer = tf.layers.dense(layer, dim,
                                activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.keras.initializers.glorot_uniform())
    output = tf.layers.dense(layer, 1, kernel_initializer=tf.contrib.keras.initializers.glorot_uniform())

    return output


def nn2(layer):
    hidden_dim = [20, 10]

    for dim in hidden_dim:
        layer = tf.layers.dense(layer, dim,
                                activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.keras.initializers.glorot_uniform())
    output = tf.layers.dense(layer, 1, kernel_initializer=tf.contrib.keras.initializers.glorot_uniform())

    return output


def nn3(layer):
    # nn3 is used in classification
    hidden_dim = [32, 64, 32]

    for dim in hidden_dim:
        layer = tf.layers.dense(layer, dim,
                                activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.keras.initializers.glorot_uniform())
    output = tf.layers.dense(layer, 2, kernel_initializer=tf.contrib.keras.initializers.glorot_uniform())
    # classification:
    output = tf.nn.softmax(output)
    return output


def nn4(layer):
    # nn4 is the resnet version:
    # 3 resnet:
    hidden_dim = [32, 64, 32]
    for dim in hidden_dim:
        layer = tf.layers.dense(layer, dim,
                                activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.keras.initializers.glorot_uniform())
        
        layer = res_block(layer, res_core)
    output = tf.layers.dense(layer, 1, kernel_initializer=tf.contrib.keras.initializers.glorot_uniform())
    return output


def nn5(layer, classes=81):
    #  classification structure:
    #  with 81 classes
    hidden_dim = [32, 64, 64, 32]

    for dim in hidden_dim:
        layer = tf.layers.dense(layer, dim,
                                activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.keras.initializers.glorot_uniform())
    output = tf.layers.dense(layer, classes, kernel_initializer=tf.contrib.keras.initializers.glorot_uniform())

    return output


def res_block(x, core_nn):
    # must assure the same dimension of the x and core_xx
    return x + core_nn(x) 


def res_core(x):
    cn = x.shape[-1].value
    init_shape = x.shape
    hidden_dim = [cn, 2*cn, cn]
    for dim in hidden_dim:
        x = tf.layers.dense(x, dim,
                            activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.keras.initializers.glorot_uniform())
    
    assert x.shape == init_shape
    return x


def main():
    input = tf.zeros([10, 1000, 3])
    out = nn4(input)


if __name__ == '__main__':
    main()
