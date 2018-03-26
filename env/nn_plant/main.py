# : Harvey Chang
# : chnme40cs@gmail.com
# main function is used to make all configurations and making results:
import tqdm
import numpy as np
import tensorflow as tf
import data.dataset as D
import network.network as nn
import train.keras_trainer as trainer
import network.nn_plant as nplant
from matplotlib import pyplot as plt


def main():
    config = dict()
    #  params for signals:
    config['file_path'] = 'data/data' 
    config['sample_num'] = 100
    config['time_step'] = 800  # 8995
    config['batch_size'] = min(20, int(0.2*config['sample_num']))
    config['noise_level'] = 0

    #  dimensions:
    config['m'] = 3  #  previous m x and y
    config['channel'] = config['m']*2 

    #  params for net structure:
    config['typ'] = 'rnn'
    config['core_nn'] = 1  #  choose inside
    config['target'] = 'action'  #  or classification

    #  params for training:
    config['diff'] = False  # start diff structure
    config['test_mode'] = False

    config['save'] = True
    config['restore'] = True
    config['first_rnn'] = True
    config['training_epochs'] = 1000
    config['learning_rate'] = 1e-2

    #  log structure:
    config['pre_log_dir'] = {'base': 'train_log/base/pre_model'}

    config['log_dir'] = {'base': 'train_log/base/{}_model'.format(config['typ'])}
    #  directory for tensorboard
    config['board_dir'] = 'train_log/log'

    #  read datas:
    dataX, dataY, val_dataX, val_dataY = D.read_data(config['file_path'], config['sample_num'], config['batch_size'])

    #  plant structure
    '''
    myplant = nplant.Nplant(config, val_dataX[0], val_dataY[0])
    myplant.restore()
    myplant.test()
    #  test
    for i in range(5):
        myplant.test(dataX[i], dataY[i], 1000)
    '''
    mytrainer = trainer.Trainer(config)
    #  add data:
    if not config['test_mode']:
        mytrainer.add_data(dataX, dataY)
    mytrainer.add_data(val_dataX, val_dataY, data_type='validation')
    #  mytrainer.train()
    #  mytrainer.test()
    mytrainer.auto_learn()


if __name__ == '__main__':
    main()
