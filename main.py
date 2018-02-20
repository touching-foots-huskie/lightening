# -*- coding:utf-8 -*-
import train.trainer as trainer


if __name__ == "__main__":
    config = dict()
    # environment:
    config['frequency'] = 100
    config['t1'] = 1
    config['plant'] = 'poly'
    config['algorithm'] = 'pid'

    #  network
    config['log_dir'] = 'train_log/log'
    config['save_dir'] = 'train_log/model'

    config['gap_len'] = 10
    config['a_lr'] = 1e-4
    config['c_lr'] = 1e-4

    config['gamma'] = 0.3
    config['epsilon'] = 0.2
    config['a_update_steps'] = 20
    config['c_update_steps'] = 20

    #  training:
    config['iter_num'] = 10
    config['episodes_per_iter'] = 20
    #  config dim_dict
    config['a_dim_dict'] = {
            'poly': 1,
            }

    config['s_dim_dict'] = {
            'poly': 6,
            }

    config['a_dim'] = config['a_dim_dict'][config['plant']]
    config['s_dim'] = config['s_dim_dict'][config['plant']]

    #  run algorithm:
    mytrainer = trainer.trainer(config)
    mytrainer.train()
