# -*- coding:utf-8 -*-
import math
import json
import random
import numpy as np
import env.env as Env
import network.ppo as NN

class trainer:
    def __init__(self, config):
        self.config = config
        self.env = Env.Env(config)
        self.nn = NN.PPO(config)
        #  get Xs and Ys:
        X, Y = self.env.run_algorithms()
        self.input_datas, self.Signals = self.env.get_list(X, Y)
        self.list_num = self.input_datas.shape[0]

    def run_episode(self, signal, input_data):
        #  X is the source:
        result = dict()
        for i, xn in enumerate(signal):
            #  preprossess xn
            state = list(input_data)
            state.append(xn)
            state = np.array(state).reshape([1, -1])

            cx = self.nn.choose_action(state).reshape([1, -1])
            xn += np.squeeze(cx)

            y, input_data = self.env.plant(input_data, xn)
            #  y is assumed to be identical to xn:
            reward = 1.0/max(abs(y - xn), 1e-6)
            #  add results:
            if i == 0:
                result['state'] = state
                result['old_act'] = cx
                result['reward'] = np.array(reward)
            else:
                result['state'] = np.concatenate((result['state'], state), axis=0)
                result['old_act'] = np.concatenate((result['old_act'], cx), axis=0)
                result['reward'] = np.append(result['reward'], reward)

        #  calculate q_val:
        r_len = result['reward'].shape[0]
        result['q_val'] = np.zeros(r_len)
        for i, _r in enumerate(result['reward'][::-1]):
            if i == 0:
                result['q_val'][r_len-i-1] = _r
            else:
                result['q_val'][r_len-i-1] = _r + self.config['gamma']*result['q_val'][r_len-i]

        #  get value and advantage:
        result['q_val'] = result['q_val'].reshape([-1, 1])
        result['value'] = self.nn.get_v(result['state']).reshape([-1, 1]) 
        result['advantage'] = result['q_val']- result['value']
        return result

    def run_policy(self):
        results = dict()
        for i in range(self.config['episodes_per_iter']):
            #  prepare data:
            num = random.randint(0, self.list_num - 1)
            input_data = self.input_datas[num]
            signal = self.Signals[num]

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
            self.nn.update(results)
            self.nn.log(results)
            print('iter:{}| reward:{}'.format(i, results['reward'].mean()))


if __name__ == "__main__":
    pass
