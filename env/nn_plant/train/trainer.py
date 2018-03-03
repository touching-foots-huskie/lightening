#: Harvey Chang
#: chnme40cs@gmail.com
import random
import numpy as np
import tensorflow as tf
import data.dataset as D
import network.network as NN
from matplotlib import pyplot as plt


class Trainer:
    def __init__(self, config):
        '''
        :param typ: type is divided into pre/rnn
        :param m: m is using m data in X
        :param n: n is using n data in Y
        :return: return a nn model
        '''
        self.config = config
        self.nn = NN.Pdn(config) 

        if self.config['restore']:
            for name in self.config['restore_parts']:
                self.nn.restore(name)

        self.batch_size = config['batch_size']
        #  m, n are the major structure
        self.m = config['m']
        self.n = config['n']

        self.max_l = max(self.m-1, self.n)
    
    def add_data(self, X, Y, V, data_type='train'):
        #  generating training data
        #  X shape(N, time_step)
        #  Y shape(N, time_step)
        dataX, dataY = [], []
        #  sequence them
        for i in range(X.shape[0]):
            if self.config['typ'] == 'pre':
                datax, datay = D.p_sequence(X[i], Y[i], V[i], self.m, self.n, self.max_l)

            elif self.config['typ'] == 'rnn':
                datax, datay = D.r_sequence(X[i], Y[i], V[i], self.m)

            dataX.append(datax)
            #  change y:
            dataY.append(datay)

        #  split Y:
        dataY = np.asarray(dataY)  #  batch, time_step
        dataYb, dataYn = D.split(dataY)
        dataYn = dataYn[:, :, np.newaxis]
        '''
        dataYn = np.clip(dataYn, -round(self.config['classes']/2.0), round(self.config['classes']/2.0))
        dataYn += round(self.config['classes']/2.0)
        '''
        dataX = np.asarray(dataX)[:, 2:]
        if data_type == 'train':
            self.train_dataX = dataX
            self.train_dataY = dataYn
            self.train_original_x = np.array(X[:, self.m:])
            #  log the shape when adding trainX:

        if data_type == 'validation':
            self.val_data = dict()
            self.val_dataX = dataX
            self.val_dataY = dataYn
            self.val_original_x = np.array(X[:, self.m:])
            self.val_data['X'] = self.val_dataX
            self.val_data['Y'] = self.val_dataY

    def train(self):
        #  begin training process:
        step = 0
        while step < self.config['training_epochs']:
            for i in range(self.config['sample_num']//self.batch_size-1):
                datax = self.train_dataX[i * self.batch_size:(i + 1) * self.batch_size]
                datay = self.train_dataY[i * self.batch_size:(i + 1) * self.batch_size]

                data = dict()
                data['X'] = datax
                data['Y'] = datay
                if self.config['append']:
                    data['Xp'] = datax
                _, t_accuracy, pred = self.nn.update(data) 

            #  validation:
            val_accuracy, pred = self.nn.validate(self.val_data)
            print('training is {}|validation: {}|step: {}'.format(t_accuracy, val_accuracy, step))
            step += 1

        if self.config['save']:
            self.nn.save()

    def test(self):
        #  begin training process:
        val_accuracy, pred = self.nn.validate(self.val_data)
        print('val loss is {}'.format(val_accuracy))
        plt.plot(pred[0, 100:200])
        plt.plot(self.val_data['Y'][0, 100:200])
        plt.show()
        
    #  drawing examination
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
        loss = np.sum((Y[:, 2:] - (2*Y[:, 1:-1] - Y[:, :-2]))**2)
        print('basic error is {}'.format(loss))


def main():
    pass

