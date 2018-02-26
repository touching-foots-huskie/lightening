#  : Harvey Chang
#  : chnme40cs@gmail.com
import tensorflow as tf
import data.dataset as D
import network.network as NN
from matplotlib import pyplot as plt


class Trainer:
    def __init__(self, config):
        '''
        :param typ: type is divided into pre/rnn
        :param m: m is using m data in X
        :param n: n is using n data in Y
        :return: return a nn model
        '''
        self.config = config
        self.nn = NN.Pdn(config) 

        if self.config['restore']:
            self.nn.restore()

        self.batch_size = config['batch_size']
        #  m, n are the major structure
        self.m = config['m']
        self.n = config['n']
        #  mp, np are the appended structure
        self.mp = config['mp']
        self.np = config['np']

        self.max_l = max(self.m-1, self.n, self.mp-1, self.np)
        if ((self.m-1) == self.max_l) or ((self.mp-1) == self.max_l):
            self.axis = 0
        elif (self.n == self.max_l) or (self.np == self.max_l):
            self.axis = 1
    
    def add_data(self, X, Y, data_type='train'):
        #  generating training data
        #  X shape(N, time_step)
        #  Y shape(N, time_step)
        dataX, dataY = [], []
        dataXp = []
        for i in range(X.shape[0]):
            if self.config['typ'] == 'pre':
                if self.config['reverse'] == 'none':
                    datax, datay = D.p_sequence(X[i], Y[i], self.m, self.n, self.max_l, self.axis)
                elif (self.config['reverse'] == 'left') or (self.config['reverse'] == 'right'):
                    datax, datay = D.p_sequence(Y[i], X[i], self.m, self.n, self.max_l, self.axis)

            elif self.config['typ'] == 'rnn':
                datax, datay = D.r_sequence(X[i], Y[i], self.m)

            dataX.append(datax)
            dataY.append(datay)

            if self.config['append']:
                if self.config['reverse'] == 'none':
                    dataxp, _ = D.p_sequence(X[i], Y[i], self.mp, self.np, self.max_l, self.axis)
                elif (self.config['reverse'] == 'left') or (self.config['reverse'] == 'right'):
                    dataxp, _ = D.p_sequence(Y[i], X[i], self.mp, self.np, self.max_l, self.axis)
                dataXp.append(dataxp)

        if data_type == 'train':
            self.train_dataX = np.asarray(dataX)
            self.train_dataY = np.asarray(dataY)[:, :, np.newaxis]
            self.train_original_x = np.array(X[:, self.m:])
            #  log the shape when adding trainX:
            if self.config['append']:
                self.train_dataXp = np.asarray(dataXp)

        if data_type == 'validation':
            self.val_data = dict()
            self.val_dataX = np.asarray(dataX)
            self.val_dataY = np.asarray(dataY)[:, :, np.newaxis]
            self.val_original_x = np.array(X[:, self.m:])
            self.val_data['X'] = self.val_dataX
            self.val_data['Y'] = self.val_dataY
            if self.config['append']:
                self.val_dataXp = np.asarray(dataXp)
                self.val_data['Xp'] = self.val_dataXp

    def train(self):
        #  begin training process:
        step = 0
        while step < self.config['training_epochs']:
            for i in range(int((1.0 * self.data_num) / self.batch_size)):
                datax = self.train_dataX[i * self.batch_size:(i + 1) * self.batch_size]
                datay = self.train_dataY[i * self.batch_size:(i + 1) * self.batch_size]

                data = dict()
                data['X'] = datax
                data['Y'] = datay
                if self.config['append']:
                    dataxp = self.train_dataXp[i * self.batch_size:(i + 1) * self.batch_size]
                    data['Xp'] = dataxp

                _, r_loss, pred = self.nn.update(data) 

            #  validation:
            val_loss, pred = self.nn.validate(self.val_data)
            print('training is {}|validation: {}| step: {}'.format(r_loss, val_loss, step))
            step += 1

        if self.config['save']:
            self.nn.save()

    def exam_result(self):
        #  begin training process:
        val_loss, pred = self.nn.validate(self.val_data)
        self.plot_val(pred)
        
    #  drawing examination
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
        plt.plot(np.squeeze(self.val_dataY[exam_num]), label='Actual Y')
        plt.plot(np.squeeze(pred[exam_num]), label='Predict Y')
        plt.plot(pred[exam_num] - self.val_dataY[exam_num], label='error')
        plt.legend(loc='upper right')
        plt.show()

def main():
    pass

