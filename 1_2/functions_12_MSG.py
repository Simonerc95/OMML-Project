import pandas as pd
from itertools import product
import numpy as np
import numexpr as ne
import time
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


def rbf_scores(X, c, v, sigma):
    X_norm = np.sum(X ** 2, axis=-1)
    c_norm = np.sum(c ** 2, axis=-1)
    return np.dot(ne.evaluate('exp( (A + B - 2 * C) / -sigma)', {
        'A': X_norm[:, None],
        'B': c_norm[None, :],
        'C': np.dot(X, c.T),
        'sigma': sigma ** 2,
    }), v)


class RBF:
    def __init__(self, df, N, rho=1e-5, sigma=1, ttv=[.7, .85], act_func='hyperbolic_tangent', random_state=1679838):

        assert N > 0, 'N must be positive!'
        assert isinstance(N, int), 'N must be an integer'
        self.N = N

        assert sigma >= 0, 'Sigma must be positive!'
        self.sigma = sigma

        assert rho >= 10 ** (-5), 'Rho too low!'
        assert rho <= 10 ** (-3), 'Rho too high!'
        self.rho = rho

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
        self.C = np.random.normal(loc=0., scale=1. / self.N, size=(self.N, self.n))  # N x n
        self.v = np.random.normal(loc=0., scale=1. / self.N, size=(self.N, 1))  # N x 1

        self.Loss_list = []

    def fit(self, method='BFGS', maxiter=2000, disp=True, print_=False):

        t = time.time()
        vec = self._to_vec()

        opt = minimize(self._optimize, vec, method=method, options=
        {'maxiter': maxiter, 'disp': disp})
        self.minimize_obj = opt
        self.C, self.v = self._to_array(opt.x)
        if print_:
            print(f'Time: {time.time() - t}')
            print(f'Loss:{self.minimize_obj["fun"]}')


    def _compute_loss(self, C, v, dataset, loss_reg=False):
        sigma = self.sigma
        rho = self.rho

        if dataset == 'valid':
            X = self.X_valid
            y = self.y_valid
        elif dataset == 'train':
            X = self.X_train
            y = self.y_train
        elif dataset == 'test':
            X = self.X_test
            y = self.y_test

        f_x = rbf_scores(X, C, v, sigma)
        Loss = np.sum((f_x.reshape(y.shape) - y) ** 2) / (2 * len(y))
        # self.Loss_list.append(Loss)

        if loss_reg:
            L2 = np.linalg.norm(np.concatenate((C, v), axis=None)) ** 2  # regularization
            Loss_reg = Loss + (rho * L2)
            return Loss_reg
        else:
            return Loss

    def predict(self, X):
        return rbf_scores(X, self.C, self.v, self.sigma)

    def _to_vec(self):
        return np.hstack([self.C.flatten(), self.v.flatten()])

    def _to_array(self, vec):
        N = self.N
        n = self.n

        assert vec.shape == (N * n + N,)
        return vec[:N * n].reshape(N, n), vec[N * n:].reshape(N, 1)

    def _optimize(self, vec, dataset='train'):
        C, v = self._to_array(vec)
        return self._compute_loss(C, v, dataset, loss_reg=True)

    def get_loss(self, loss_type):
        out = {}
        if loss_type == 'all':
            for type_ in ('train', 'valid', 'test',):
                out[type_] = self._compute_loss(self.C, self.v, dataset=type_)
        else:
            out[loss_type] = self._compute_loss(self.C, self.v, dataset=loss_type)

        return out


params = {
    'N_vals': list(range(20, 50, 1)),
    'sigma_vals': np.arange(.5, 1.5, .1),
    'rho_vals': [1e-5, 1e-4, 1e-3]}


def random_search(model, df, params, iterations=100, seed=1679838, print_=False):
    np.random.seed(seed)
    combinations = np.array(list(product(*params.values())))

    np.random.shuffle(combinations)
    combinations = combinations[:iterations]
    assert iterations <= len(combinations), 'iterations exceeded number of combinations'
    t = time.time()                                # x[0] = N, x[1] = sigma, x[2] = rho
    res = np.apply_along_axis(lambda x: get_opt(model, int(x[0]), x[1], x[2], df), 1, combinations)
    print(f"Total time: {time.time() - t}")
    best_loss = np.inf
    idx = None
    for i, row in enumerate(res):
        if row['loss'] < best_loss:
            idx = i
            best_loss = row['loss']
    print('Best loss:', res[idx]['loss'], '\nBest params:', res[idx]['param'])

    return res[idx]


def get_opt(model, n, sigma, rho, df):
    print_ = True
    best_loss = np.inf
    best_params = {}

    network = model(df=df, N=n, rho=rho, sigma=sigma)
    network.fit(print_=print_)
    current_loss = network._compute_loss(network.C, network.v, 'valid')
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
    plt.savefig('out_12', dpi=100)
