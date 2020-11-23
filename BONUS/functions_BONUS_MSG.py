import pandas as pd
from itertools import product
import numpy as np
import numexpr as ne
import time
from scipy.optimize import least_squares as ls
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from joblib import Parallel, delayed
import pickle


def rbf_scores(X, c, sigma):
    X_norm = np.sum(X ** 2, axis=-1)
    c_norm = np.sum(c ** 2, axis=-1)
    return ne.evaluate('exp( (A + B - 2 * C) / -sigma)', {
        'A': X_norm[:, None],
        'B': c_norm[None, :],
        'C': np.dot(X, c.T),
        'sigma': sigma ** 2
    })


class RBF:
    __name__ = 'Radial Basis Function Network'
    def __init__(self, df, N, rho=1e-5, sigma=1, sigma_C=2, ttv=[.8], act_func='hyperbolic_tangent', random_state=1679838):

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
        self.train_data, self.valid_data = np.split(df.sample(frac=1),[int(ttv[0] * len(df))])

        self.X_train = self.train_data.iloc[:, :2].to_numpy()  # P x n
        self.y_train = self.train_data.iloc[:, 2].to_numpy()  # P x 1
        self.X_valid = self.valid_data.iloc[:, :2].to_numpy()
        self.y_valid = self.valid_data.iloc[:, 2].to_numpy()

        self.n = self.X_train.shape[1]
        self.C = np.random.normal(loc=0., scale=2. / self.N, size=(self.N, self.n))  # N x n
        self.v = np.random.normal(loc=0., scale=2. / self.N, size=(self.N, 1))  # N x 1
        self.C_temp = self.C
        self.sigma_C = sigma_C
        self.Loss_list = []


    def fit(self, method='BFGS', maxiter=2000, disp=True, print_=True):
        np.random.RandomState(self.seed)
        t = time.time()
        vec = self._to_vec()

        opt = minimize(self._optimize, vec, method=method, options=
        {'maxiter': maxiter, 'disp': disp})
        self.minimize_obj = opt
        self.C, self.v = self._to_array(opt.x)
        if print_:
            print(f'Time: {time.time() - t}')
            print(f'Loss:{self.minimize_obj["fun"]}')
            
            
    def extreme_learning(self, num_iter=10000, print_=False):
        best_loss = np.inf
        t = time.time()
        for c in range(num_iter):
            self.C_temp = np.random.normal(scale=self.sigma_C, size=(self.N, self.n))  # N x n 
            try:
                v = self.opt_v()
                current_loss = self._optimize(v)
                if current_loss < best_loss:
                    self.C = self.C_temp
                    self.v = v
                    best_loss = current_loss
            except:
                print(f'bad parameter combination: N={self.N}, sigma={self.sigma}, rho={self.rho}, sigma_C={self.sigma_C}')
                break
        if print_:
            print(f'Total time for Extreme learning: {time.time()-t}')
            print(f'Best loss: {best_loss}')
        self._get_all_loss()  
    

    def _compute_loss(self, C, v, dataset, loss_reg=False):

        sigma = self.sigma
        rho = self.rho

        if dataset == 'valid':
            X = self.X_valid
            y = self.y_valid
        elif dataset == 'train':
            X = self.X_train
            y = self.y_train


        g_x = rbf_scores(X, C, sigma)
        f_x = g_x.dot(v)
        Loss = np.sum((f_x.reshape(y.shape) - y) ** 2) / (2 * len(y))
        # self.Loss_list.append(Loss)

        if loss_reg:
            L2 = np.linalg.norm(v) ** 2  # regularization
            Loss_reg = Loss + (rho * L2)
            return Loss_reg
        else:
            return Loss
        
        

    def opt_v(self):
        g_x = rbf_scores(self.X_train, self.C_temp, self.sigma)
        return np.dot(np.linalg.pinv(g_x), self.y_train)

    def predict(self, X):
        return rbf_scores(X, self.C, self.sigma).dot(self.v)

    def predict_and_rmse_test(self, test_df):
        X_bonus = test_df.iloc[:, :2].to_numpy()  # P x n
        y_bonus = test_df.iloc[:, 2].to_numpy()  # P x 1
        y_pred = self.predict(X=X_bonus)
        MSE = np.sum((y_pred.reshape(y_bonus.shape) - y_bonus) ** 2) / (2 * len(y_bonus))
        print(f'Mean Squared Error on new test data is: {MSE}')

    def predict_and_rmse_base(self, test_df):
        X_bonus = test_df.iloc[:, :2].to_numpy()  # P x n
        y_bonus = test_df.iloc[:, 2].to_numpy()  # P x 1
        y_pred = self.predict(X=X_bonus)
        MSE = np.sum((y_pred.reshape(y_bonus.shape) - y_bonus) ** 2) / (2 * len(y_bonus))
        print(f'Mean Squared Error on train data is: {MSE}')

    def _to_vec(self):
        return np.hstack([self.C.flatten(), self.v.flatten()])

    def _to_array(self, vec):
        N = self.N
        n = self.n

        assert vec.shape == (N * n + N,)
        return vec[:N * n].reshape(N, n), vec[N * n:].reshape(N, 1)

    def _optimize(self, v, dataset='train'):
        C = self.C_temp
        return self._compute_loss(C, v, dataset, loss_reg=True)

    def get_loss(self, loss_type):
        out = {}
        if loss_type == 'all':
            for type_ in ('train', 'valid',):
                out[type_] = self._compute_loss(self.C, self.v, dataset=type_)
        else:
            out[loss_type] = self._compute_loss(self.C, self.v, dataset=loss_type)

        return out

    def _get_all_loss(self):
        self.train_loss = self._compute_loss(self.C, self.v, dataset='train', loss_reg=False)
        self.valid_loss = self._compute_loss(self.C, self.v, dataset='valid', loss_reg=False)
        self.train_loss_reg = self._compute_loss(self.C, self.v, dataset='train', loss_reg=True)

    def print_params(self):
        print('\nModel selected:', self.__name__,
              '\nBest N:', self.N,
              '\nBest rho:', self.rho,
              '\nBest sigma:', round(self.sigma, 1))

        
        
params = {
    'N_vals': list(range(20, 80, 1)),
    'sigma_vals': np.arange(.3, 1.5, .05),
    'rho_vals': [1e-5, 1e-4, 5e-4, 1e-3],
    'sigma_C': np.arange(.5, 3, .1)
    }


def random_search(model, df, params, iterations=10000, seed=1679838, print_=True, n_jobs=-1):
    np.random.seed(seed)
    combinations = np.array(list(product(*params.values())))
    np.random.shuffle(combinations)
    combinations = combinations[:iterations]
    
    t = time.time()                                # x[0] = N, x[1] = sigma, x[2] = rho
    res = Parallel(n_jobs=n_jobs, verbose=10)\
        (delayed(get_opt)(model, int(x[0]), x[1], x[2], x[3], df, print_) for x in combinations)
    print(f"\nTotal time: {time.time() - t}")
    best_loss = np.inf
    model = None
    for mod in res:
        if mod.valid_loss < best_loss:
            best_loss = mod.valid_loss
            model = mod
    with open('best_model_bonus.pkl', 'wb') as pkl:
        pickle.dump(model, pkl)
    return model


def get_opt(model, n, sigma, rho, sigma_C, df, print_=True):
    network = model(df=df, N=n, rho=rho, sigma=sigma, sigma_C=sigma_C)
    network.extreme_learning()
    return network


def get_plot(net, df):

    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)

    x_1, x_2 = np.meshgrid(x, y)
    x_1 = x_1.flatten()  # .reshape(-1,1)
    x_2 = x_2.flatten()  # .reshape(-1,1)
    x_ = np.vstack([x_1, x_2])
    z_ = net.predict(x_.T)
    #print(z_.shape)

    fig = plt.figure(figsize=(8, 6))

    ax = Axes3D(fig)
    x_1, x_2 = np.meshgrid(x, y)
    ax.plot_surface(x_1, x_2, z_.reshape(x_1.shape), rstride=1, cstride=1,
                    cmap='gist_rainbow_r', edgecolor='none')
    ax.set_title('surface')
    plt.savefig('out_bonus.png', dpi=100)
    

