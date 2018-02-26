# -*- coding:utf-8 -*-
import math
import tqdm
import numpy as np
import scipy.io as scio


class Env:
    def __init__(self, config):
        self.config = config
        self.plant = self.plant_wrapper(config['plant'])

        self.init_x = np.zeros(self.x_num)
        self.init_y = np.zeros(self.y_num)
        self.input_dim = self.x_num + self.y_num
        #  get signals:
        self.get_signals()
    
    #  manipulations：
    def reset(self, x_data, y_data):
        self.init_x = x_data
        self.init_y = y_data

    #  plants configurations:
    def plant_wrapper(self, plant_choice):
        if plant_choice == 'poly':
            #  data configurations:
            self.x_num = 2
            self.y_num = 3
            return self.poly_plant
        elif plant_choice == 'triangle':
            self.x_num = 2
            self.y_num = 3
            return self.sin_plant

    def poly_plant(self, input_data, xn=0):
        y1, y2, y3, x1, x = input_data
        y = (y1*y2*y3*x1*(y3 - 1.0) + x)/(1.0 + y2**2 + y3*2)
        ni_data = [y, y1, y2, x, xn]
        return y, ni_data

    def sin_plant(self, input_data, xn=0):
        y1, y2, y3, x1, x = input_data
        y = math.sin(y1) + math.cos(y2) + math.sin(y3) + x1 + x 
        ni_data = [y, y1, y2, x, xn]
        return y, ni_data
        
    def get_list(self, X, Y):
        #  when getting a long X and long Y, return Xs, and Ys
        #  X Y should be nparray
        Xs = []
        Ys = []
        Signals = []
        assert X.shape == Y.shape
        start_loc = max(self.x_num, self.y_num)
        for i in tqdm.tqdm(range(start_loc, X.shape[0] - self.config['gap_len'])):
            Xs.append(X[(i - self.x_num + 1):(i + 1)])
            Ys.append(Y[(i - self.y_num):i])
            Signals.append(X[i:(i + self.config['gap_len'])])
        print('data generation finished!')
        input_datas = np.concatenate((Xs, Ys), axis=-1)
        return input_datas, Signals

    #  run algorithms:  
    #  pid:
    def run_algorithms(self):
        if self.config['algorithm'] == 'pid':
            return self.pid(self.config['source_signal'])
        else:
            print('No such algorithm!')

    def pid(self, signal_type='sin'):
        #  config for pid:
        Kp = 1.0
        
        #  prepare for data
        if signal_type == 'sin':
            X = self.sin_signal
        elif signal_type == 'cos':
            X = self.cos_signal
        elif signal_type == 'rgs':
            X = self.rgs_signal

        Y = []
        _error = 0
        input_data = np.concatenate((self.init_y, self.init_x))
        _input_data = input_data 
        for xn in X:
            xn += _error * Kp
            y, input_data = self.plant(input_data, xn)
            _error = input_data[-1] - y
            print(_error)
            Y.append(y)

        Y = np.array(Y)
        #  after running the pid, save the signal and start:
        self.test_signal = X
        self.test_input = _input_data
        return X, Y 

    def iter_learn(self,input_data):
        #  one step target iter_learning:
        #  iter_learn may not converge, we need to sort it.
        xc = 0
        xt = input_data[-1]
        data_lists = []
        for i in range(self.config['learn_iter']):
            I_input_data = input_data
            #  add compensation:
            I_input_data[-1] += xc
            y, _ = self.plant(I_input_data)
            #  update compensation:
            _error = abs(xt - y)
            xc += _error * 0.1
            #  add xc, _error into batch
            data_lists.append([xc, _error])
        data_lists = np.array(data_lists)
        print(data_lists)
        data_lists = data_lists[np.argsort(data_lists[:, 1])]
        bxc, berror = data_lists[0]
        return bxc, berror

    #  different signals:
    def get_signals(self):
        t = np.linspace(0, self.config['t1'], (self.config['t1'] * self.config['frequency']) + 1)
        #  sin signals:
        self.sin_signal = np.sin(t)
        #  cos signals:
        self.cos_signal = np.cos(t)-1.0
        #  rgs signals: times 10
        self.rgs_signal = scio.loadmat('env/signal/rgs.mat')['y'][:, 0].reshape([-1])[:1000]*10
        
        self.test_input = np.concatenate((self.init_y, self.init_x))

    def choose_signal(self):
        if self.config['source_signal'] == 'sin':
            return self.sin_signal

        elif self.config['source_signal'] == 'cos':
            return self.cos_signal
        
        elif self.config['source_signal'] == 'rgs':
            return self.rgs_signal
