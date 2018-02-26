# -*- coding:utf-8 -*-
import queue
import math
import json
import random
import numpy as np
import env.env as Env
import network.dpo as NN
from matplotlib import pyplot as plt 


class trainer:
    def __init__(self, config):
        self.config = config
        self.env = Env.Env(config)
        self.nn = NN.DPO(config)
        if self.config['restore']:
            self.nn.restore()
            print('model has been restored!')
        #  get Xs and Ys:
        X = self.env.choose_signal()
        self.input_datas, self.Signals = self.env.get_list(X, X)
        self.list_num = self.input_datas.shape[0]
        self.results_list = queue.deque(maxlen=self.config['max_batches'])
    
    def batch_explore(self):
        results = dict()
        for i in range(self.config['explore_batch']):
            num = random.randint(0, self.list_num - 1)
            input_data = self.input_datas[num]
            input_data += np.random.randn(input_data.shape[0])*self.config['noise_level']
            #  explore using iter_learn
            bxc, berror = self.env.iter_learn(input_data)   
            #  reshape structure
            breward = np.array(self.reward(berror)).reshape([1, -1])
            bxc = bxc.reshape([1, -1])
            berror = berror.reshape([1, -1])
            input_data = np.array(input_data).reshape([1, -1])

            if i == 0:
                results['state'] = input_data
                results['target_action'] = bxc
                results['berror'] = berror
                results['breward'] = breward
            else:
                results['state'] = np.concatenate((results['state'], input_data), axis=0)
                results['target_action'] = np.concatenate((results['target_action'], bxc), axis=0)
                results['berror'] = np.concatenate((results['berror'], berror), axis=0)
                results['breward'] = np.concatenate((results['breward'], breward), axis=0)
        self.results_list.append(results)

    def batch_train(self):
        #  find a batch outof the results_list
        r_list_num = len(self.results_list)
        num = random.randint(0, r_list_num-1)
        t_results = self.results_list[num]
        for i in range(self.config['explore_batch']):
            input_data = t_results['state'][i]
            xt = input_data[-1]
            xc = np.squeeze(self.nn.choose_action(input_data.reshape([1, -1])))
            input_data[-1] += xc
            y, _ = self.env.plant(input_data)
            _error = abs(xt - y)
            reward = self.reward(_error)
            #  exam:
            #  reshape
            advantage = (t_results['breward'][i] - reward).reshape([1, -1])
            _error = np.array(_error).reshape([1, -1])

            if i == 0:
                print('xc: {}, bxc: {}'.format(xc, t_results['target_action'][i]))
                print('error: {}, berror: {}'.format(_error, t_results['berror'][i]))
                print('reward: {}, breward: {}'.format(reward, t_results['breward'][i]))
                print(advantage)

            if i == 0:
                t_results['advantage'] = advantage
                t_results['error'] = _error
            else:
                t_results['advantage'] = np.concatenate((t_results['advantage'], advantage), axis=0)
                t_results['error'] = np.concatenate((t_results['error'], _error), axis=0)

        #  after update the advantage part:
        #  train the structure:
        self.nn.update(t_results)
        self.nn.log(t_results)
        print('in this iter |error:{}'.format(t_results['error'].mean()))

    def train(self):
        #  train is to arrage batch_explore and batch_train
        if self.config['restore']:
            self.nn.restore()
            print('model has been restored!')

        #  explore first
        self.batch_explore()
        for i in range(self.config['iter_num']):
            self.batch_train()

        #  save after train:
        if self.config['save']:
            self.nn.save()
            print('model has been saved!')

    def test(self, signal_type='sin', test_len=1):
        # test need to redesign 
        pass

    #  reward design
    def reward(self, x):
        #  alpha, beta, x0, parameters:
        alpha = 1.0
        beta = 1.0
        x0 = 0.1
        y0 = 10.0
        y1 = -10.0

        if x<x0:
            r = y0*(math.exp(-alpha*x) - math.exp(-alpha*x0))
        else:
            r = -beta*(x - x0) 
        return max(y1, r)


if __name__ == "__main__":
    pass
