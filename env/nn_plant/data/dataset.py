# : Harvey Chang
# : chnme40cs@gmail.com
#  this file is used to describe the non-linear plant for learning:
import pdb
import tqdm
import random
import numpy as np
import scipy.io as scio
from matplotlib import pyplot as plt
from sklearn.utils import shuffle


def read_mat(data_num, filename):
    #  different scales in data:
    data = scio.loadmat(filename)
    if 'x' in data.keys():
        data = data['x']
        data = data * 1.0
    elif 'y' in data.keys():
        data = data['y']
        data = data
    else:
        print('wrong data structure.')

    _, total_num = data.shape
    assert total_num >= data_num
    choose_data = data[:,:data_num]
    choose_data = np.transpose(choose_data, [1, 0])
    return choose_data


def read_data(file_path, data_num, config):

    dataX = read_mat(data_num, '{}/x.mat'.format(file_path))
    dataY = read_mat(data_num, '{}/y.mat'.format(file_path))
    print('data shape is {}'.format(dataX.shape))
    #  rescale into [-1, 1]
    dataX = dataX*config['x_scale']  #  experimentally result
    dataY = dataY*config['y_scale']
    #  get shuffled 
    dataX, dataY = shuffle(dataX, dataY)
    #  get training and get validation set:
    return dataX, dataY


def r_sequence(X, Y, config):
    #  used in rnn version:
    #  read and differential
    datax, datay = [], []
    for j in range(0, Y.shape[0]-config['m']):
        datax.append(X[j:j+config['m']])
        datay.append(Y[j+config['m']-1])

    datax = np.asarray(datax, dtype=np.float32)
    datay = np.asarray(datay, dtype=np.float32)
    #  specific structure defined later
    #  differential structure
    datavm = datax[1:] - datax[:-1] 
    dataa = (datavm[1:] - datavm[:-1]) 
    datav = (datax[2:] - datax[:-2])
    dataj = dataa[2:] - dataa[:-2]
    #  clip:
    dataa = dataa[1:-1]
    datav = datav[1:-1]
    # 
    datav = datav*config['v_scale']
    dataa = dataa*config['a_scale']
    dataj = dataj*config['j_scale']
    
    # shape of new datax in 4 shorter [2:-2]
    datax = np.concatenate([datav, dataa, dataj], axis=-1)
    datay = datay[2:-2].reshape([-1, 1])

    return datax, datay


def p_sequence(X, Y, look_up):
    #  used in point version
    #  adapt to different dimensions:
    datax, datay = [], []
    for j in range(1, Y.shape[0] - look_up):
        # to predict yj
        datax.append(np.concatenate((X[j:j+look_up].reshape([-1, 1]),Y[(j-1):(j+look_up-1)].reshape([-1, 1])), axis=0))
        datay.append(Y[j+look_up-1])

    datax = np.asarray(datax, dtype=np.float32)  
    datay = np.asarray(datay, dtype=np.float32) 
    #  squeeze:
    datax = np.squeeze(datax)
    datay = np.squeeze(datay)
    return datax, datay


def s_sequence(X, look_up):
    #  generating x cascade structure
    datax = []
    for j in range(X.shape[0] - look_up):
        datax.append(X[j:j+look_up])

    datax = np.asarray(datax, dtype=np.float32)
    #  specific structure defined later
    return datax


def diff_generate(X, config):
    #  get differ structure X
    dataX = []
    for i in range(X.shape[0]):
        datax = s_sequence(X[i], config['m'])

        #  differential structutre:|x, v, a
        datavm = datax[1:] - datax[:-1] 
        dataa = (datavm[1:] - datavm[:-1]) 
        datav = (datax[2:] - datax[:-2])
        dataj = dataa[2:] - dataa[:-2]
        #  clip:
        dataa = dataa[1:-1]
        datav = datav[1:-1]
        # 
        datav = datav*config['v_scale']
        dataa = dataa*config['a_scale']
        dataj = dataj*config['j_scale']
        
        # shape of new datax in 4 shorter [2:-2]
        datax = np.concatenate([datav, dataa, dataj], axis=-1)

        dataX.append(datax)
    #  cast:
    dataX = np.asarray(dataX)
    return dataX


if __name__ == "__main__":
    pass

