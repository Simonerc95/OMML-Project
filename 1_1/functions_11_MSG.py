# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 12:45:53 2020

@author: m.broglio
"""
import pandas as pd
import numpy as np
import time
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from itertools import product



def hyperbolic_tangent(x, sigma=1):
    return (np.exp(2*sigma*x) - 1) / (np.exp(2*sigma*x) + 1)

activations = {
    'hyperbolic_tangent': hyperbolic_tangent,
    }

class MLP:
    def __init__(self, df, N, rho, sigma=1, ttv=[.7,.85], act_func='hyperbolic_tangent', random_state=1679838):
        
        assert N > 0, 'N must be positive!'
        assert isinstance(N, int), 'N must be an integer'
        self.N = N
        
        assert sigma >= 0, 'Sigma must be positive!'
        self.sigma = sigma
        
        assert rho >= 10**(-5), 'Rho too low!'
        assert rho <= 10**(-3), 'Rho too high!'
        self.rho = rho
        
        assert act_func in activations.keys(), f'"{act_func}" is a not supported activation function'  
        self.activation = activations[act_func]
        
        self.seed = random_state
        self.neurons = []
        np.random.seed(self.seed)
        self.train_data, self.test_data, self.valid_data = np.split(df.sample(frac=1), [int(ttv[0]*len(df)), int(ttv[1]*len(df))])
        
        self.X_train = self.train_data.iloc[:, :2].to_numpy() # P x n 
        self.y_train = self.train_data.iloc[:, 2].to_numpy() # P x 1
        self.X_valid = self.valid_data.iloc[:, :2].to_numpy() 
        self.y_valid = self.valid_data.iloc[:, 2].to_numpy() 
        self.X_test = self.test_data.iloc[:, :2].to_numpy() 
        self.y_test = self.test_data.iloc[:, 2].to_numpy()
        
        self.n = self.X_train.shape[1]
        self.W = np.random.normal(scale=2, size=(self.N, self.n)) # N x n
        self.v = np.random.normal(scale=2, size=(self.N, 1)) # N x 1
        self.b = np.random.normal(scale=2, size=(self.N, 1)) # N x 1
        self.Loss_list = []
        
        
    def fit(self, method = 'bfgs', maxiter=1000, print_=True):
        disp = print_
        t = time.time()
        vec = self._to_vec()
            
        opt = minimize(self._optimize, vec, method=method, options =
                       {'maxiter':maxiter, 'disp':disp})
        self.minimize_obj = opt
        self.W, self.v, self.b = self._to_array(opt.x)
        
        self._get_all_loss()
        self.fit_time = f'Fit time: {time.time() - t}'
        if print_:
            print(f'Time: {self.fit_time}')
            print(f'Loss_train_reg_fit from minimize:{self.minimize_obj["fun"]}')
            print(f'Loss_valid :{self.valid_loss}')
            
    def _compute_loss(self, W, v, b, dataset, loss_reg=False):
        sigma = self.sigma
        rho = self.rho
        activation = self. activation
        
        if dataset == 'valid':
            X = self.X_valid
            y = self.y_valid
        elif dataset == 'train':
            X = self.X_train
            y = self.y_train 
        elif dataset == 'test':
            X = self.X_test
            y = self.y_test             
            
        xx = W.dot(X.T) - b # N x P Ogni colonna è l'output degli N neuroni per una specifica x (unità statistica)
        g_x = activation(xx, sigma) #, sigma=self.sigma) # N x P
        f_x = g_x.T.dot(v) # P x 1 - g_x.T = P x N @ N x 1
        Loss = np.sum((f_x.reshape(y.shape) - y)**2) / (2 * len(y))
        # self.Loss_list.append(Loss)
        
        if loss_reg:
            L2 =  np.linalg.norm(np.concatenate((W, v, b), axis=None))**2    # regularization
            Loss_reg = Loss + (rho * L2)
            return Loss_reg
        else:
            return Loss
    
    def predict(self, X):
        xx = self.W.dot(X.T) - self.b # N x P Ogni colonna è l'output degli N neuroni per una specifica x (unità statistica)
        g_x = self.activation(xx) #, sigma=self.sigma) # N x P
        f_x = g_x.T.dot(self.v) # P x 1 - g_x.T = P x N @ N x 1
               
        return f_x
    
    def _to_vec(self):
        return np.hstack([self.W.flatten(), self.v.flatten(), self.b.flatten()])
    
    def _to_array(self, vec):
        N = self.N
        n = self.n
      
        assert vec.shape == (N * n + ( 2 * N),)
        return vec[:N*n].reshape(N,n), vec[N*n:-N].reshape(N,1), vec[-N:].reshape(N,1)
           
    def _optimize(self, vec, dataset = 'train'): 
        W, v, b = self._to_array(vec)
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
    
    def print_loss_param(self, time=True):
        if time:
            print(f'\nFit Time: ', self.fit_time)
        print('\nBest N :', self.N,
          '\nBest rho :', self.rho,
          '\nBest sigma :', self.sigma,
          '\nBest train_loss: ', self.train_loss,
          '\nBest valid_loss: ', self.valid_loss,
          '\nBest test_loss: ', self.test_loss,)
 
params = {
    'N_vals': list(range(1, 30, 1)),
    'sigma_vals': np.arange(.5, 1.5, .1),
    'rho_vals': [1e-5, 1e-3, 1e-4]}


def random_search(model, df, params, iterations=40, seed=1679838, print_=True, n_jobs=-1):
    np.random.seed(seed)
    combinations = np.array(list(product(*params.values())))
    np.random.shuffle(combinations)
    combinations = combinations[:iterations]
    assert iterations <= len(combinations), 'iterations exceeded number of combinations'
    t = time.time()                                # x[0] = N, x[1] = sigma, x[2] = rho
    res = Parallel(n_jobs=n_jobs, verbose=10)\
        (delayed(get_opt)(model, int(x[0]), x[1], x[2], df, print_) for x in combinations)
    print(f"\nTotal time: {time.time() - t}")
    best_loss = np.inf
    model = None
    for mod in res:
        if mod.valid_loss < best_loss:
            model = mod

    return model


def get_opt(model, n, sigma, rho, df, print_=True):
    network = model(df=df, N=n, rho=rho, sigma=sigma)
    network.fit(print_=print_)
    return network


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
    plt.savefig('out_11', dpi=100)