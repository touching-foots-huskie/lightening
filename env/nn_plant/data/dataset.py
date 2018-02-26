# : Harvey Chang
# : chnme40cs@gmail.com
# this file is used to describe the non-linear plant for learning:
import tqdm
import random
import numpy as np
import scipy.io as scio
from matplotlib import pyplot as plt


def read_mat(data_num, filename):
    data = scio.loadmat(filename)
    if 'y' in data.keys():
        data = data['y']  # it only has one value
    else:
        data = data['Out']
    _, total_num = data.shape
    assert total_num >= data_num
    choose_data = data[:,:data_num]
    choose_data = np.transpose(choose_data, [1, 0])
    return choose_data


def read_data(file_path, tail_name, data_num):
    #  return trainXY, valXY
    dataX = read_mat(data_num, '{}/train_{}'.format(file_path, tail_name))
    dataY = read_mat(data_num, '{}/out_train_{}'.format(file_path, tail_name))
    val_dataX = read_mat(data_num, '{}/val_{}'.format(file_path, tail_name))
    val_dataY = read_mat(data_num, '{}/out_val_{}'.format(file_path, tail_name))
    return dataX, dataY, val_dataX, val_dataY


def r_sequence(X, Y, look_up):
    #  used in rnn version:
    datax, datay = [], []
    for j in range(Y.shape[0] - look_up):
        datax.append(X[j:j + look_up])
        datay.append(Y[j + look_up])
    datax = np.asarray(datax, dtype=np.float32)
    datay = np.asarray(datay, dtype=np.float32)
    return datax, datay


def p_sequence(X, Y, m=2, n=3, max_l=3, axis=0):
    #  used in pre_version
    #  m is the lookup in x
    #  n is the lookup in y

    datax, datay = [], []
    if axis==1:
        for j in range(max_l, Y.shape[0]):
            datax.append(np.concatenate([X[j-m+1:j+1], Y[j-n:j]]))
            datay.append(Y[j])
    elif axis==0:
        for j in range(max_l, Y.shape[0]):
            datax.append(np.concatenate([X[j-m+1:j], Y[j-n:j]]))
            datay.append(Y[j])

    datax = np.asarray(datax, dtype=np.float32)  # (997, 5)
    datay = np.asarray(datay, dtype=np.float32)  # (997, 1)
    return datax, datay


if __name__ == "__main__":
    pass

