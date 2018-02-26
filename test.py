# -*- coding:utf-8 -*-
#  this test is used to test env
import numpy as np
import env.env as env
import train.trainer as trainer
from matplotlib import pyplot as plt


if __name__ == "__main__":
    config = dict()
    # environment:
    config['learn_iter'] = 20
    config['frequency'] = 100
    config['t1'] = 1
    config['plant'] = 'triangle'
    config['algorithm'] = 'pid'
    config['source_signal'] = 'sin'
    config['noise_level'] = 0.0

    #  run algorithm:
    myenv = env.Env(config) 
    input_data = np.array([1.2, 1.1, 1.0, 1.18, 1.3])
    xc, _error = myenv.iter_learn(input_data)
    print('xc: {}| error: {}'.format(xc, _error))

