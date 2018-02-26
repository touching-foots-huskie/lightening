# : Harvey Chang
# : chnme40cs@gmail.com
# main function is used to make all configurations and making results:
import tqdm
import plant
import Models
import core_nn
import app_funcs
import numpy as np
import tensorflow as tf


def main():
    config = dict()
    # params for signals:
    config['sample_num'] = 100
    config['ptype'] = 'sin'  # if sin use previous structure
    config['No'] = 1  # choose plant
    config['noise_level'] = 0
    config['data_read'] = True  # if True, the program will directly read result from file

    # params for net structure:
    config['typ'] = 'rnn'
    config['target'] = 'action'  # or classification
    # reverse version:
    config['reverse'] = 'none'  # left right none
    config['core_nn'] = 1  # choose inside
    config['m'] = 4  # previous m x # in reverse version we reverse the m and n
    config['n'] = 4  # previous n y

    # append structure
    config['append'] = False
    config['app_func'] = 2  # choose app structure
    config['mp'] = 2  # appended mp x
    config['np'] = 3  # appended np y

    # params for training:
    config['test_mode'] = True
    config['restore'] = True
    config['first_rnn'] = True
    config['batch_size'] = min(30, int(0.2*config['sample_num']))
    config['training_epochs'] = 100
    config['learning_rate'] = 0.1

    config['pre_log_dir'] = 'train_log/typ_pre_cn_{}_plant{}_{}_{}_{}_{}'.\
        format(config['core_nn'], config['No'],
               config['m'], config['n'], config['mp'], config['np']
               )

    config['log_dir'] = 'train_log/typ_{}_cn_{}_plant{}_{}_{}_{}_{}'.\
        format(config['typ'], config['core_nn'], config['No'],
               config['m'], config['n'], config['mp'], config['np']
               )

    '''
       # get data:
    if not config['data_read']:
        if not config['test_mode']:
            dataX, dataY = plant.m_data_gen(config['sample_num'],
                                        config['ptype'],
                                        No=config['No'], noise_level=config['noise_level'])
            np.save('dataset/trainX_{}_{}_{}.npy'.format(config['sample_num'], config['No'], config['ptype']), dataX)
            np.save('dataset/trainY_{}_{}_{}.npy'.format(config['sample_num'], config['No'], config['ptype']), dataY)

        val_dataX, val_dataY = plant.m_data_gen(config['batch_size'],
                                                config['ptype'],
                                                No=config['No'], noise_level=config['noise_level'])

    else:
        # if data read: 
    '''

    if not config['test_mode']:
        # dataX = np.load('dataset/trainX_{}_{}_{}.npy'.format(config['sample_num'], config['No'], config['ptype']))
        # dataY = np.load('dataset/trainY_{}_{}_{}.npy'.format(config['sample_num'], config['No'], config['ptype']))
        dataX = plant.m_signal(config['sample_num'], 'signal_data/realData/train_{}'.format(config['ptype']))
        dataY = plant.m_signal(config['sample_num'], 'signal_data/realData/out_train_{}'.format(config['ptype']))
    val_dataX =  plant.m_signal(config['batch_size'], 'signal_data/realData/val_{}'.format(config['ptype']))
    val_dataY =  plant.m_signal(config['batch_size'], 'signal_data/realData/out_val_{}'.format(config['ptype']))

    print('val_dataX average: {}'.format(np.mean(val_dataX)))
    print('data loaded!')

    myModel = Models.Model(typ=config['typ'], core_nn=core_nn.nn_wrapper(config['core_nn']), config=config,
                           app_func=app_funcs.app_wrapper(config['app_func']))

    if not config['test_mode']:
        myModel.add_data(dataX, dataY)
    else:
        myModel.add_data(val_dataX, val_dataY)
    myModel.add_data(val_dataX, val_dataY, data_type='validation')

    # establish network:
    myModel.network()
    if not config['test_mode']:
        myModel.train()
    else:
        myModel.exam_result()


if __name__ == '__main__':
    main()
