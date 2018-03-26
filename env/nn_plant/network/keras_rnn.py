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


class keras_model:
    def __init__(self, config):
        #  configuration:
        self.config = config
        #  define model
        self.model = Sequential()
        self.model.add(CuDNNGRU(32, input_shape=(self.config['time_step'], 3*self.config['m'],), return_sequences=True))
        self.model.add(CuDNNGRU(64, return_sequences=True))
        self.model.add(CuDNNGRU(64, return_sequences=True))
        self.model.add(CuDNNGRU(64, return_sequences=True))
        self.model.add(CuDNNGRU(32, return_sequences=True))
        self.model.add(CuDNNGRU(1, return_sequences=True))

        #  define learning structure
        self.model.compile(loss='mse', optimizer='adagrad', metrics=['mae'])


    def train(self, dataX, dataY):
        self.model.fit(dataX, dataY, batch_size=self.config['batch_size'], nb_epoch=self.config['training_epochs'], verbose=2, validation_split=0.2)

    def save(self):
        self.model.save_weights('train_log/model.h5')

    def restore(self):
        self.model.load_weights('train_log/model.h5')

    def validate(self, val_dataX, val_dataY):
        predict_Y = self.model.predict(val_dataX)
        plt.plot(predict_Y[0], label='prediction')
        plt.plot(val_dataY[0], label='actual')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
