# -*- coding:utf-8 -*-
import tqdm
import numpy as np


class Env:
    def __init__(self, config):
        self.config = config
        self.plant = self.plant_wrapper(config['plant'])

        self.init_x = np.zeros(self.x_num)
        self.init_y = np.zeros(self.y_num)
        self.input_dim = self.x_num + self.y_num
    
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

    def poly_plant(self, input_data, xn):
        y1, y2, y3, x1, x = input_data
        y = (y1*y2*y3*x1*(y3 - 1.0) + x)/(1.0 + y2**2 + y3*2)
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
            return self.pid()
        else:
            print('No such algorithm!')

    def pid(self):
        #  config for pid:
        Kp = 1.0
        t = np.linspace(0, self.config['t1'], (self.config['t1'] * self.config['frequency']) + 1)
        X = np.sin(t)
        Y = []
        _error = 0
        input_data = np.concatenate((self.init_y, self.init_x))
        for xn in X:
            xn += _error * Kp
            y, input_data = self.plant(input_data, xn)
            _error = input_data[-1] - y
            Y.append(y)

        Y = np.array(Y)
        return X, Y 

