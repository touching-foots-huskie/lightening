#  keras model is used in trainers.| after trainer added data 
import pdb
import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import CuDNNLSTM
from keras.layers import CuDNNGRU
from keras.optimizers import Adam
from keras.initializers import Orthogonal


class keras_model:
    def __init__(self, config):
        #  configuration:
        self.config = config
        #  define model
        #  gru structure
        self.model = Sequential()
        init = Orthogonal()

        self.model.add(CuDNNGRU(32, input_shape=(None, 3*self.config['m'],), return_sequences=True, \
        kernel_initializer=init))
        self.model.add(CuDNNGRU(64, return_sequences=True, kernel_initializer=init))
        self.model.add(CuDNNGRU(64, return_sequences=True, kernel_initializer=init))
        self.model.add(CuDNNGRU(64, return_sequences=True, kernel_initializer=init))
        self.model.add(CuDNNGRU(32, return_sequences=True, kernel_initializer=init))
        self.model.add(CuDNNGRU(1, return_sequences=True, kernel_initializer=init))
        
        adam = Adam(lr=config['learning_rate'])
        #  define learning structure
        self.model.compile(loss='mse', optimizer=adam, metrics=['mae'])


    def train(self, dataX, dataY):
        self.model.fit(dataX, dataY, batch_size=self.config['batch_size'], nb_epoch=self.config['training_epochs'], verbose=2, validation_split=0.2)

    def save(self):
        self.model.save_weights('train_log/model.h5')

    def restore(self):
        self.model.load_weights('train_log/model.h5')

    def validate(self, val_dataX, val_dataY):
        print('Start validate!')
        print(val_dataX.shape)
        predict_Y = self.model.predict(val_dataX)
        #  plt.plot(val_dataX[-1, :, 0], label='V')
        #  plt.plot(val_dataX[-1, :, 3], label='A')
        #  plt.plot(val_dataX[-1, :, 6], label='J')
        plt.plot(predict_Y[-1], label='prediction')
        plt.plot(val_dataY[-1], label='actual')
        plt.legend()
        plt.show()

    def implement(self, im_dataX):
        predict_Y = self.model.predict(im_dataX)
        return predict_Y


if __name__ == '__main__':
    main()
