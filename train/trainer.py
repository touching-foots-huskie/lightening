# -*- coding:utf-8 -*-
import math
import json
import random
import numpy as np
import env.env as Env
import network.ppo as NN
from matplotlib import pyplot as plt 


class trainer:
    def __init__(self, config):
        self.config = config
        self.env = Env.Env(config)
        self.nn = NN.PPO(config)
        if self.config['restore']:
            self.nn.restore()
            print('model has been restored!')
        #  get Xs and Ys:
        X = self.env.choose_signal()
        self.input_datas, self.Signals = self.env.get_list(X, X)
        self.list_num = self.input_datas.shape[0]

    def run_episode(self, signal, input_data):
        #  X is the source:
        result = dict()
        for i, xn in enumerate(signal[:-1]):
            #  preprossess xn
            state = list(input_data)
            state.append(xn)
            state.append(signal[i+1])
            state = np.array(state).reshape([1, -1])

            cx = self.nn.choose_action(state).reshape([1, -1])
            xc = np.squeeze(cx) + xn

            y, input_data = self.env.plant(input_data, xc)
            #  y is assumed to be identical to xn:
            reward = self.reward(abs(y-xn))
            _error = abs(y-xn)
            #  add results:
            if i == 0:
                result['state'] = state
                result['old_act'] = cx
                result['error'] = _error
            elif i == 1:
                result['reward'] = reward
            #  in shortest structure we don't need more.
            '''
                result['state'] = np.concatenate((result['state'], state), axis=0)
                result['old_act'] = np.concatenate((result['old_act'], cx), axis=0)
                result['error'] = np.append(result['error'], _error)
            else:
                result['state'] = np.concatenate((result['state'], state), axis=0)
                result['old_act'] = np.concatenate((result['old_act'], cx), axis=0)
    
                result['reward'] = np.append(result['reward'], reward)
                result['error'] = np.append(result['error'], _error)

            '''
        #  calculate q_val:
        #  the reward of the final action is 0
        result['reward'] = np.append(result['reward'], 0)
        r_len = result['reward'].shape[0]
        result['q_val'] = np.zeros(r_len)
        for i, _r in enumerate(result['reward'][::-1]):
            if i == 0:
                result['q_val'][r_len-i-1] = _r
            else:
                result['q_val'][r_len-i-1] = _r + self.config['gamma']*result['q_val'][r_len-i]

        #  get value and q_val:
        result['q_val'] = result['q_val'].reshape([-1, 1])
        result['value'] = self.nn.get_v(result['state']).reshape([-1, 1]) 

        return result

    def run_policy(self):
        results = dict()
        for i in range(self.config['episodes_per_iter']):
            #  prepare data:
            num = random.randint(0, self.list_num - 1)
            input_data = self.input_datas[num]
            signal = self.Signals[num]

            #  add noise to input_data:
            input_data += np.random.randn(input_data.shape[0])*self.config['noise_level']

            result = self.run_episode(signal, input_data) 

            if i == 0:
                results = result
            else:
                for key, value in result.items():
                    results[key] = np.concatenate((results[key], value), axis=0)

        return results

    def train(self):
        for i in range(self.config['iter_num']):
            #  update the environment first
            results = self.run_policy()
            #  normalize in results:
            results['q_val'] = (results['q_val'] - results['q_val'].mean())/results['q_val'].std()
            results['advantage'] = results['q_val'] - results['value']
            results['advantage'] = (results['advantage'] - results['advantage'].mean())/results['advantage'].std()

            self.nn.update(results)
            self.nn.log(results)
            print('iter:{}| reward:{}'.format(i, results['reward'].mean()))
            print(results['old_act'][:10])
        #  save after train:
        if self.config['save']:
            self.nn.save()
            print('model has been saved!')

    def test(self, signal_type='sin', test_len=1):
        #  test the policy in test_len * gap_len:
        test_len *= self.config['gap_len']
        if signal_type == 'sin':
            results = self.run_episode(self.env.sin_signal[:test_len], self.env.test_input)
        elif signal_type == 'cos':
            results = self.run_episode(self.env.cos_signal[:test_len], self.env.test_input)
        elif signal_type == 'rgs':
            results = self.run_episode(self.env.rgs_signal[100:100+test_len], self.env.test_input)

        print('Test| reward:{}'.format(results['reward'].mean()))

        print(results['error'])
        print(results['reward'])
        print(results['state'])
        #  reward design:

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
