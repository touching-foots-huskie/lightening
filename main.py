# -*- coding:utf-8 -*-


if __name__ == "__main__":
    config = dict()
    # environment:
    config['frequency'] = 100
    config['t1'] = 1
    config['plant'] = 'poly'

    #  network
    config['a_lr'] = 1e-4
    config['c_lr'] = 1e-4

    config['epsilon'] = 0.2
    config['a_update_steps'] = 20
    config['c_update_steps'] = 20

    #  training:
    config['episode_per_iter'] = 20
    #  config dim_dict
    config['a_dim_dict'] = {
            'poly': 1,
            }

    config['s_dim_dict'] = {
            'poly': 8,
            }

    config['a_dim'] = config['a_dim_dict'][config['plant']]
    config['s_dim'] = config['s_dim_dict'][config['plant']]

