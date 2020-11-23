# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 12:45:53 2020

@author: m.broglio
"""
import time
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


def hyperbolic_tangent(x, sigma=1):
    return (np.exp(2 * sigma * x) - 1) / (np.exp(2 * sigma * x) + 1)


activations = {
    'hyperbolic_tangent': hyperbolic_tangent,
}


class MLP:
    def __init__(self, df, N, rho, sigma=1, ttv=[.7, .85], act_func='hyperbolic_tangent', random_state=1679838):

        assert N > 0, 'N must be positive!'
        assert isinstance(N, int), 'N must be an integer'
        self.N = N

        assert sigma >= 0, 'Sigma must be positive!'
        self.sigma = sigma

        assert rho >= 10 ** (-5), 'Rho too low!'
        assert rho <= 10 ** (-3), 'Rho too high!'
        self.rho = rho

        assert act_func in activations.keys(), f'"{act_func}" is a not supported activation function'
        self.activation = activations[act_func]

        self.seed = random_state
        self.neurons = []
        np.random.seed(self.seed)
        self.train_data, self.test_data, self.valid_data = np.split(df.sample(frac=1),
                                                                    [int(ttv[0] * len(df)), int(ttv[1] * len(df))])

        self.X_train = self.train_data.iloc[:, :2].to_numpy()  # P x n
        self.y_train = self.train_data.iloc[:, 2].to_numpy()  # P x 1
        self.X_valid = self.valid_data.iloc[:, :2].to_numpy()
        self.y_valid = self.valid_data.iloc[:, 2].to_numpy()
        self.X_test = self.test_data.iloc[:, :2].to_numpy()
        self.y_test = self.test_data.iloc[:, 2].to_numpy()

        self.n = self.X_train.shape[1]
        self.W = np.zeros((self.N, self.n))  # N x n
        self.v = np.zeros((self.N, 1))  # N x 1
        self.b = np.zeros((self.N, 1))  # N x 1
        self.W_tmp = self.W  # N x n
        self.b_tmp = self.b  # N x 1
        self.Loss_list = []



    def extreme_learning(self, num_iter=100):
        self.max_iter = num_iter
        best_loss = 1000
        self.tot_fun = 0
        self.tot_grad = 'gradient free'
        self.method = 'llsp'
        t = time.time()
        for c in range(num_iter):
            self.W_tmp = np.random.normal(scale=2.3, size=(self.N, self.n))
            self.b_tmp = np.random.normal(scale=2.5, size=(self.N, 1))
            v = self.opt_v()
            self.tot_fun += 1
            current_loss = self._optimize(v)
            if current_loss < best_loss:
                self.W = self.W_tmp
                self.b = self.b_tmp
                self.v = v
                best_loss = current_loss
                self._get_all_loss()
        
        self.fit_time = time.time() - t
        
        # print(f'Total time for Extreme learning: {time.time() - t}')
        # print(f'Best Regularized Loss: {best_loss}')

    def _compute_loss(self, W, v, b, dataset, loss_reg=False):
        sigma = self.sigma
        rho = self.rho
        activation = self.activation

        if dataset == 'valid':
            X = self.X_valid
            y = self.y_valid
        elif dataset == 'train':
            X = self.X_train
            y = self.y_train
        elif dataset == 'test':
            X = self.X_test
            y = self.y_test


        xx = W.dot(X.T) - b  # N x P Ogni colonna è l'output degli N neuroni per una specifica x (unità statistica)
        g_x = activation(xx, sigma)  # , sigma=self.sigma) # N x P
        f_x = g_x.T.dot(v)  # P x 1 - g_x.T = P x N @ N x 1
        Loss = np.sum((f_x.reshape(y.shape) - y) ** 2) / (2 * len(y))
        # self.Loss_list.append(Loss)

        if loss_reg:

            L2 = np.linalg.norm(v)**2  # regularization
            Loss_reg = Loss + (rho * L2)
            return Loss_reg
        else:
            return Loss

    def opt_v(self):
        xx = self.W_tmp.dot(self.X_train.T) - self.b_tmp #N x P
        g_x = self.activation(xx, self.sigma) #N x P
        return np.dot(np.linalg.pinv(g_x.T), self.y_train)

    def predict(self, X):
        xx = self.W.dot(
            X.T) - self.b  # N x P Ogni colonna è l'output degli N neuroni per una specifica x (unità statistica)
        g_x = self.activation(xx)  # , sigma=self.sigma) # N x P
        f_x = g_x.T.dot(self.v)  # P x 1 - g_x.T = P x N @ N x 1

        return f_x

    def _to_vec(self):
        return np.hstack([self.W.flatten(), self.v.flatten(), self.b.flatten()])

    def _to_array(self, vec):
        N = self.N
        n = self.n

        assert vec.shape == (N * n + (2 * N),)
        return vec[:N * n].reshape(N, n), vec[N * n:-N].reshape(N, 1), vec[-N:].reshape(N, 1)

    def _optimize(self, v, dataset='train'):
        W = self.W_tmp
        b = self.b_tmp
        return self._compute_loss(W, v, b, dataset, loss_reg=True)

    def get_loss(self, loss_type):
        out = {}
        if loss_type == 'all':
            for type_ in ('train', 'valid', 'test',):
                out[type_] = self._compute_loss(self.W, self.v, self.b, dataset=type_)
        else:
            out[loss_type] = self._compute_loss(self.W, self.v, self.b, dataset=loss_type)

        return out

    def _get_all_loss(self):
        self.train_loss = self._compute_loss(self.W, self.v, self.b, dataset='train', loss_reg=False)
        self.valid_loss = self._compute_loss(self.W, self.v, self.b, dataset='valid', loss_reg=False)
        self.test_loss = self._compute_loss(self.W, self.v, self.b, dataset='test', loss_reg=False)
        self.train_loss_reg = self._compute_loss(self.W, self.v, self.b, dataset='train', loss_reg=True)

    def print_loss_params(self, time=True):
        # if time:
        #     print(f'\n', self.fit_time)
        print('\nBest N :', self.N,
          '\nBest sigma :', self.sigma,
          '\nBest rho :', self.rho,
          '\nExtreme learning iterations:', self.max_iter,
          '\nOptimization method:', self.method,
          '\nNumber of function evaluations:', self.tot_fun,
          '\nNumber of gradient evaluations:', self.tot_grad,
          '\nTime for optimizing the network:', self.fit_time,
          '\nBest train_loss: ', self.train_loss,
          # '\nBest valid_loss: ', self.valid_loss,
          '\nBest test_loss: ', self.test_loss,)


def get_opt(model, n, sigma, rho, df):
    print_ = True
    best_loss = np.inf
    best_params = {}

    perceptron = model(df=df, N=n, rho=rho, sigma=sigma)
    perceptron.fit(print_=print_)
    current_loss = perceptron._compute_loss(perceptron.W, perceptron.v, perceptron.b, 'valid')
    if current_loss < best_loss:
        best_params['N'] = n
        best_params['rho'] = rho
        best_params['sigma'] = sigma
        best_loss = current_loss
    return {'param': best_params, 'loss': best_loss}


def get_loss(model, loss_type):
    if loss_type == 'train':
        return model.get_loss('train')
    elif loss_type == 'valid':
        return model.get_loss('valid')
    elif loss_type == 'test':
        return model.get_loss('test')
    elif loss_type == 'all':
        return model.get_loss('all')
    else:
        raise Exception('loss type not supported!')


def get_plot(net):
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)

    x_1, x_2 = np.meshgrid(x, y)
    x_1 = x_1.flatten()  # .reshape(-1,1)
    x_2 = x_2.flatten()  # .reshape(-1,1)
    x_ = np.vstack([x_1, x_2])
    z_ = net.predict(x_.T)

    fig = plt.figure(figsize=(8, 6))

    ax = plt.axes(projection='3d')
    x_1, x_2 = np.meshgrid(x, y)
    ax.plot_surface(x_1, x_2, z_.reshape(x_1.shape), rstride=1, cstride=1,
                    cmap='gist_rainbow_r', edgecolor='none')
    ax.set_title('surface')
    plt.savefig('out_21', dpi=100)
