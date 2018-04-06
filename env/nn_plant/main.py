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
    #  scales:
    config['x_scale'] = 10;    #  x should multiply x_scale
    config['y_scale'] = 5e4;
    config['v_scale'] = 50;
    config['a_scale'] = 5000;
    config['j_scale'] = 1e5;
    #  params for signals:
    config['file_path'] = 'data/data' 
    config['sample_num'] = 1218
    config['val_num'] = 23
    config['time_step'] = 990  # in mat is 1000, we choose an integer
    config['batch_size'] = 128
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

    config['save'] = True
    config['restore'] = False
    config['training_epochs'] = 150
    config['learning_rate'] = 1e-3

    #  log structure:
    config['pre_log_dir'] = {'base': 'train_log/base/pre_model'}

    config['log_dir'] = {'base': 'train_log/base/{}_model'.format(config['typ'])}
    #  directory for tensorboard
    config['board_dir'] = 'train_log/log'

    #  read datas:
    dataX, dataY = D.read_data(config['file_path'], config['sample_num'], config)
    val_dataX, val_dataY = D.read_data('data/test', config['val_num'], config)

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
    mytrainer.add_data(dataX, dataY)
    mytrainer.add_data(val_dataX, val_dataY, data_type='validation')
    #  mytrainer.add_data(dataX, dataY, data_type='validation')
    mytrainer.train()
    #  mytrainer.test()
    #  mytrainer.auto_learn()
    #  mytrainer.implement()


if __name__ == '__main__':
    main()
