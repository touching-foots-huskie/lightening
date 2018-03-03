# : Harvey Chang
# : chnme40cs@gmail.com
#  this file is used to describe the non-linear plant for learning:
import tqdm
import random
import numpy as np
import scipy.io as scio
from matplotlib import pyplot as plt
from sklearn.utils import shuffle


def read_mat(data_num, filename):
    data = scio.loadmat(filename)
    if 'x' in data.keys():
        data = data['x']
        data = data * 1.0
    elif 'y' in data.keys():
        data = data['y']
        data = data * 10.0
    elif 'v' in data.keys():
        data = data['v']
        data = data * 10.0
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
    dataV = read_mat(data_num, '{}/v.mat'.format(file_path))
    #  get shuffled 
    dataX, dataY, dataV = shuffle(dataX, dataY, dataV)
    #  get training and get validation set:

    return dataX[val_num:], dataY[val_num:], dataV[val_num:],\
            dataX[:val_num], dataY[:val_num], dataV[:val_num]


def r_sequence(X, Y, V, look_up):
    #  used in rnn version:
    datax, datay = [], []
    for j in range(Y.shape[0] - look_up):
        datax.append(np.concatenate((X[j:j + look_up], V[j:j + look_up]), axis=0))
        datay.append(Y[j + look_up])
    datax = np.asarray(datax, dtype=np.float32)
    datay = np.asarray(datay, dtype=np.float32)
    return datax, datay


def p_sequence(X, Y, V, m, n, max_l):
    #  used in pre_version
    #  m is the lookup in x
    #  n is the lookup in y

    datax, datay = [], []
    for j in range(max_l, Y.shape[0]-1):
        datax.append(np.concatenate((1e2*(X[j-m+2:j+1] - X[j-m+1:j]), X[j:j+1],\
                1e4*(Y[j-n+1:j] - Y[j-n:j-1]), Y[j-1:j])))

        datay.append(Y[j])

    datax = np.asarray(datax, dtype=np.float32)  
    datay = np.asarray(datay, dtype=np.float32) 
    return datax, datay


def split(Y):
    #  split Y into one degree and noise:
    Yb = 2*Y[:, 1:-1] - Y[:, :-2]
    Yn = Y[:, 2:] - (2*Y[:, 1:-1] - Y[:, :-2])
    #  balance the noise
    return Yb, Yn*1e5


def one_hot(Y, classes):
    #  Y is in the shape of [N, M, 1]
    Y_shape = Y.shape
    Y_shape[-1] = classes
    Yh = np.zeros(Y_shape)
    


if __name__ == "__main__":
    pass

