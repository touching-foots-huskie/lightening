# : Harvey Chang
# : chnme40cs@gmail.com
# main function is used to make all configurations and making results:
import tqdm
import numpy as np
import tensorflow as tf
import data.dataset as D
import network.network as nn
import train.trainer as trainer


def main():
    config = dict()
    #  params for signals:
    config['signal'] = 'sin'  
    config['file_path'] = 'signal_data/realData' 
    config['batch_size'] = 20
    config['sample_num'] = 100
    config['time_step'] = 1000

    #  dimensions:
    config['m'] = 4  #  previous m x # in reverse version we reverse the m and n
    config['n'] = 4  #  previous n y
    config['channel'] = config['m'] + config['n'] 

    #  params for net structure:
    config['typ'] = 'rnn'
    config['core_nn'] = 1  #  choose inside
    config['target'] = 'action'  #  or classification

    #  reverse version:
    config['reverse'] = 'none'  #  left right none

    #  append structure
    config['append'] = False
    config['app_func'] = 2  #  choose app structure
    config['mp'] = 4  #  appended mp x
    config['np'] = 4  #  appended np y
    config['p_channel'] = config['mp'] + config['np']

    #  params for training:
    config['test_mode'] = False

    config['save'] = False
    config['restore'] = False
    config['first_rnn'] = True
    config['batch_size'] = min(30, int(0.2*config['sample_num']))
    config['training_epochs'] = 100
    config['learning_rate'] = 0.1

    #  log structure:
    config['pre_log_dir'] = 'train_log/pre_model'
    config['log_dir'] = 'train_log/{}_model'.format(config['typ'])

    dataX, dataY, val_dataX, val_dataY = D.read_data(config['file_path'], config['signal'], config['sample_num'])
    mytrain = trainer.Trainer(config)
    #  add data:
    if not config['test_mode']:
        mytrain.add_data(dataX, dataY)
    mytrain.add_data(val_dataX, val_dataY, data_type='validation')


if __name__ == '__main__':
    main()
