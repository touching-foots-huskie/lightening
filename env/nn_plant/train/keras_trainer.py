#: Harvey Chang
#: chnme40cs@gmail.com
import pdb
import random
import numpy as np
import tensorflow as tf
import data.dataset as D
import network.keras_rnn as NN
from tpot import TPOTRegressor 
from matplotlib import pyplot as plt


class Trainer:
    def __init__(self, config):
        self.config = config
        '''
        self.nn = NN.keras_model(config) 

        if self.config['restore']:
            self.nn.restore()

        '''
        self.m = config['m']
        self.batch_size = config['batch_size']

    def add_data(self, X, Y, data_type='train'):
        #  generating training data
        #  X shape(N, time_step, dimension)
        #  Y shape(N, time_step, dimension)
        dataX, dataY, init_states = [], [], []
        #  sequence them
        for i in range(X.shape[0]):
            if self.config['typ'] == 'pre':
                datax, datay = D.p_sequence(X[i], Y[i], self.m)

            elif self.config['typ'] == 'rnn':
                datax, datay, init_state = D.r_sequence(X[i], Y[i], self.m)
                init_states.append(init_state)

            #  assert datax.shape(2) = config['time_step']+1
            assert datax.shape[0] >= (self.config['time_step']+3)
            
            #  differential structutre:|x, v, a
            datavm = datax[1:] - datax[:-1] 
            dataa = datavm[1:] - datavm[:-1] 
            datav = datax[2:] - datax[:-2]
            #
            datax = datax[1:self.config['time_step']+1]
            datav = datav[:self.config['time_step']]*50
            dataa = dataa[:self.config['time_step']]*5000
            datax = np.concatenate([datax, datav, dataa], axis=-1)

            #  differential structure:
            if not self.config['diff']:
                datay = datay[1:self.config['time_step']+1]
            else:
                datay = datay[1:self.config['time_step']+1] - datay[:self.config['time_step']]

            dataX.append(datax)
            #  change y:
            dataY.append(datay)
        #  cast:
        dataX = np.asarray(dataX)
        dataY = np.asarray(dataY)
        print('max in dataX:{}'.format(dataX.max()))
        print('max in dataY:{}'.format(dataY.max()))
        init_states = np.asarray(init_states)

        if len(dataY.shape) == 2:
            dataY = dataY[:, :, np.newaxis]
        #  shape config:

        #  rnn and fc is different:
        #  fc need to be shuffled and rnn need to keep in sequence

        if data_type == 'train':
            self.train_dataX = dataX
            self.train_dataY = dataY
            self.train_init_states = init_states
            self.train_original_x = np.array(X[:, self.m:])
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

    def train(self):
        self.nn.train(self.train_dataX, self.train_dataY)
        if self.config['save']:
            self.nn.save()

    def test(self):
        #  drawing examination
        self.nn.validate(self.val_dataX, self.val_dataY)

    def auto_learn(self):
        #  auto learning best feature:
        self.pipline_optimizer = TPOTRegressor(generations=5, population_size=20, cv=5, random_state=42, verbosity=2)
        self.pipline_optimizer.fit(self.train_dataX, self.train_dataY)
        print(self.pipline_optimizer.score(self.val_dataX, self.val_dataY))
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

