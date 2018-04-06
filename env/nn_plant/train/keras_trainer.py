#: Harvey Chang
#: chnme40cs@gmail.com
import pdb
import random
import numpy as np
import scipy.io as scio
import tensorflow as tf
import data.dataset as D
import network.keras_rnn as NN
from tpot import TPOTRegressor 
from matplotlib import pyplot as plt


class Trainer:
    def __init__(self, config):
        self.config = config
        self.nn = NN.keras_model(config) 

        if self.config['restore']:
            self.nn.restore()

        self.m = config['m']
        self.batch_size = config['batch_size']

    def add_data(self, X, Y, data_type='train'):
        #  generating training data
        #  X shape(N, time_step, dimension)
        #  Y shape(N, time_step, dimension)
        dataX, dataY, init_states = [], [], []
        #  sequence them
        for i in range(X.shape[0]):
            datax, datay = D.r_sequence(X[i], Y[i], self.config)
            dataX.append(datax)
            dataY.append(datay)

        #  cast:
        dataX = np.asarray(dataX)
        dataY = np.asarray(dataY)

        if len(dataY.shape) == 2:
            dataY = dataY[:, :, np.newaxis]

        if data_type == 'train':
            self.train_dataX = dataX
            self.train_dataY = dataY
            self.train_init_states = init_states
            self.train_original_x = np.array(X[:, self.m:])
            np.save('data/fast_data/trainX.npy', dataX)
            np.save('data/fast_data/trainY.npy', dataY)
            #  log the shape when adding trainX:

        if data_type == 'validation':
            self.val_data = dict()
            self.val_dataX = dataX
            self.val_dataY = dataY
            self.val_init_states = init_states
            self.val_original_x = np.array(X[:, self.m:])
            self.val_data['X'] = self.val_dataX
            self.val_data['Y'] = self.val_dataY
            self.val_data['init'] = self.val_init_states
            np.save('data/fast_data/valX.npy', dataX)
            np.save('data/fast_data/valY.npy', dataY)

    def train(self):
        self.nn.train(self.train_dataX, self.train_dataY)
        if self.config['save']:
            self.nn.save()

    def test(self):
        #  drawing examination
        self.nn.validate(self.val_dataX, self.val_dataY)

    def implement(self, iter_time=1):
        #  imdataX is the data come from target structure:
        #  imdatax should be (N, time_step, 9)
        im_dataX = scio.loadmat('data/im_data.mat')['x']*self.config['x_scale']  #  experiemental scaled
        im_dataX = im_dataX.T
        for i in range(iter_time):
            d_im_dataX = D.diff_generate(im_dataX, self.config)
            predict_Y = self.nn.implement(d_im_dataX)
            #  shape structure
            predict_Y = np.squeeze(predict_Y)
            #  padd:
            predict_Y = np.concatenate([np.zeros([2+self.config['m'],]), predict_Y, np.zeros([2,])])
            #  scale:
            predict_Y = predict_Y/self.config['y_scale']
            #  add compensation:| repeat this process
            im_dataX += predict_Y
        #  reshape into standard shape
        predict_Y = predict_Y.T

        scio.savemat('data/pre_data.mat', {'yp':predict_Y})
        return predict_Y

    def auto_learn(self):
        #  auto learning best feature:
        self.pipline_optimizer = TPOTRegressor(generations=5, population_size=20, cv=5, random_state=42, verbosity=2)
        #  process data:
        dataX = self.train_dataX.reshape([-1, 3*self.config['m']])
        dataY = self.train_dataY.reshape([-1])
        val_dataX = self.val_dataX.reshape([-1, 3*self.config['m']])
        val_dataY = self.val_dataY.reshape([-1])
        self.pipline_optimizer.fit(dataX, dataY)
        print(self.pipline_optimizer.score(val_dataX, val_dataY))
        self.pipline_optimizer.export('tpot_exported_pipeline.py')

    def distribution_draw(self, pred):
        # draw distribution:
        plt.subplot(211)
        for i in range(self.config['batch_size']):
            plt.scatter(self.val_dataX[i, :, -1], pred[i, :])

        plt.subplot(212)
        for i in range(self.config['batch_size']):
            plt.scatter(self.val_dataX[i, :, -1], self.val_dataY[i, :, -1])
        plt.show()

    def plot_val(self, pred):
        exam_num = random.randint(0, self.batch_size-1)
        #  plt.plot(np.squeeze(pred[exam_num]), label='Predict Y')
        for i in range(self.batch_size):
            plt.hist(np.squeeze(self.val_dataY[i]), label='Actual Y')
        #  plt.plot(pred[exam_num] - self.val_dataY[exam_num], label='error')
            plt.legend(loc='upper right')
            plt.show()

    def basic_error(self, Y):
        #  Y should be (N, N, 1)
        #  basic error is the error by basic moving:
        loss = np.sum((Y[:,1:] - Y[:,0:-1])**2)
        print('basic error is {}'.format(loss))


def main():
    pass

