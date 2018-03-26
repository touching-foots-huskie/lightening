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


def read_data(file_path, data_num=100, val_num=20):
    dataX = read_mat(data_num, '{}/x.mat'.format(file_path))
    dataY = read_mat(data_num, '{}/y.mat'.format(file_path))
    print('data shape is {}'.format(dataX.shape))
    #  rescale into [-1, 1]
    dataX = dataX/np.absolute(dataX).max()
    dataY = dataY/np.absolute(dataY).max()
    #  get shuffled 
    dataX, dataY = shuffle(dataX, dataY)
    #  get training and get validation set:

    return dataX[val_num:], dataY[val_num:],\
            dataX[:val_num], dataY[:val_num]


def r_sequence(X, Y, look_up):
    #  used in rnn version:
    datax, datay = [], []
    for j in range(1, Y.shape[0] - look_up):
        datax.append(X[j:j+look_up])
        datay.append(Y[j+look_up-1])

    datax = np.asarray(datax, dtype=np.float32)
    datay = np.asarray(datay, dtype=np.float32)
    #  specific structure defined later
    init_state = Y[1:look_up+1]
    return datax, datay, init_state


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


if __name__ == "__main__":
    pass

