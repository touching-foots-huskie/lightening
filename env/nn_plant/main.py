# : Harvey Chang
# : chnme40cs@gmail.com
# main function is used to make all configurations and making results:
import tqdm
import numpy as np
import tensorflow as tf
import data.dataset as D
import network.network as nn
import train.trainer as trainer
import network.nn_plant as nplant
from matplotlib import pyplot as plt


def main():
    config = dict()
    #  params for signals:
    config['file_path'] = 'data/data' 
    config['sample_num'] = 600
    config['time_step'] = 9994
    config['classes'] = 81
    config['batch_size'] = min(20, int(0.2*config['sample_num']))

    #  dimensions:
    config['m'] = 2  #  previous m x # in reverse version we reverse the m and n
    config['n'] = 3  #  previous n y
    config['channel'] = config['m'] + config['n'] 

    #  params for net structure:
    config['typ'] = 'pre'
    config['core_nn'] = 1  #  choose inside
    config['target'] = 'action'  #  or classification

    #  append structure
    config['append'] = False
    config['app_func'] = 2  #  choose app structure

    #  params for training:
    config['test_mode'] = False

    config['save'] = True
    config['restore'] = True
    config['restore_parts'] = ['base']
    config['update_part'] = 'base'  #  base, append, all 
    config['first_rnn'] = True
    config['training_epochs'] = 1000
    config['learning_rate'] = 1e-1

    #  log structure:
    config['pre_log_dir'] = {'base': 'train_log/base/pre_model',
                             'append': 'train_log/append/pre_model'}

    config['log_dir'] = {'base': 'train_log/base/{}_model'.format(config['typ']),
                        'append': 'train_log/append/{}_model'.format(config['typ'])}

    dataX, dataY, dataV, val_dataX, val_dataY, val_dataV = D.read_data(config['file_path'], config['sample_num'], config['batch_size'])

    myplant = nplant.Nplant(config)
    myplant.restore()
    for i in range(5):
        myplant.test(dataX[i], dataY[i], 1000)
    
    '''
    mytrainer = trainer.Trainer(config)
    #  add data:
    if not config['test_mode']:
        mytrainer.add_data(dataX, dataY, dataV)
    mytrainer.add_data(val_dataX, val_dataY, val_dataV, data_type='validation')
    #  mytrainer.train()
    mytrainer.test()
    '''

if __name__ == '__main__':
    main()
