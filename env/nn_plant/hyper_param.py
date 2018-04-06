#  keras model is used in trainers.| after trainer added data 
import pdb
import numpy as np
from matplotlib import pyplot as plt
from keras.initializers import Orthogonal
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import CuDNNGRU
from keras.layers import GRU
from keras.optimizers import Adam

#  autokeras:
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional


def data():
    # collectting data structure:
    x_train = np.load('data/fast_data/trainX.npy')
    y_train = np.load('data/fast_data/trainY.npy')
    x_test = np.load('data/fast_data/valX.npy')
    y_test = np.load('data/fast_data/valY.npy')
    #  look shape
    print('x_train shape: {}'.format(x_train.shape))
    print('x_test shape: {}'.format(x_test.shape))
    return x_train, y_train, x_test, y_test


def create_model(x_train, y_train, x_test, y_test):
    model = Sequential()
    init = Orthogonal() 
    #  dropout_rate = {{choice([0, 0.1, 0.3, 0.5])}}
    model.add(CuDNNGRU(32, input_shape=(None, 9,), return_sequences=True, \
              kernel_initializer=init))

    model.add(CuDNNGRU(64, return_sequences=True, kernel_initializer=init))
    model.add(CuDNNGRU(64, return_sequences=True, kernel_initializer=init))
    model.add(CuDNNGRU(64, return_sequences=True, kernel_initializer=init))

    model.add(CuDNNGRU(32, return_sequences=True, kernel_initializer=init))
    model.add(CuDNNGRU(1, return_sequences=True, kernel_initializer=init))
    #  define learning structure
    adam = Adam(lr={{choice([1e-1, 1e-2, 1e-3])}})
    model.compile(loss={{choice(['mae', 'mse'])}}, metrics=['mae'], optimizer=adam)
    #  find best structure in 100 epochs
    model.fit(x_train, y_train, batch_size={{choice([32, 64, 128])}}, epochs=150, verbose=2, validation_data=(x_test, y_test))
    score, maerror = model.evaluate(x_test, y_test, verbose=0)
    print('Test mae:', maerror)
    return {'loss': maerror, 'status': STATUS_OK, 'model': model}
    

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model, \
                                        data=data, \
                                        algo=tpe.suggest, \
                                        max_evals=5,\
                                        trials=Trials())

    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
