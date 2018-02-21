# -*- coding:utf-8 -*-
import numpy as np
import env.env as env
import train.trainer as trainer
from matplotlib import pyplot as plt


if __name__ == "__main__":
    config = dict()
    # environment:
    config['frequency'] = 100
    config['t1'] = 1
    config['plant'] = 'poly'
    config['algorithm'] = 'pid'
    config['source_signal'] = 'sin'
    config['noise_level'] = 0.0

    #  network
    config['log_dir'] = 'train_log/log'
    config['save_dir'] = 'train_log/model'

    config['gap_len'] = 2  #  shortest structure.
    config['a_lr'] = 1e-3
    config['c_lr'] = 1e-3

    config['gamma'] = 0.3
    config['epsilon'] = 0.1
    config['a_update_steps'] = 20
    config['c_update_steps'] = 20

    #  training:
    config['save'] = True
    config['restore'] = False
    config['iter_num'] = 1000
    config['episodes_per_iter'] = 20
    #  config dim_dict
    config['a_dim_dict'] = {
            'poly': 1,
            }

    config['s_dim_dict'] = {
            'poly': 7,
            }

    config['a_dim'] = config['a_dim_dict'][config['plant']]
    config['s_dim'] = config['s_dim_dict'][config['plant']]

    #  run algorithm:
    mytrainer = trainer.trainer(config)
    mytrainer.train() 

    '''
    x = np.linspace(0,1, 100)
    r = [mytrainer.reward(_x) for _x in x]
    plt.plot(x, r)
    plt.show()

    '''
