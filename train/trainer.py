# -*- coding:utf-8 -*-
import json
import random
import numpy as np
import env.env as Env
import network.nn as NN

class trainer:
    def __init__(self, config):
        self.config = config
        self.env = Env(config)
        self.nn = NN(config)
        self.Xs = []
        self.input_datas = []

    def read_list(self):
        #  read the data for input data:
        pass

    def run_episode(self, X, input_data):
        #  X is the source:
        for xn in X:
            #  preprossess xn
            y, input_data = myenv.plant(input_data, xn)
            Y.append(y)
        Y = np.array(Y)
        return Y

    def run_policy(self):
        results = dict()
        for i in range(config['episode_per_iter']):
            num = random.randint(0, self.list_num)
            X = self.Xs[num]
            input_data = self.input_datas[num]
            Y = self.run_episode(X, input_data) 

        return results

