from functions_21_MSG import *
np.random.seed(1679838)
df = pd.read_csv(r'..\OMML2020_Assignment_1_Dataset.csv')
params = {'N': 16, 'rho': 1e-05, 'sigma': 0.9}
model = MLP(df=df, **params)
model.extreme_learning(num_iter=30000)
loss = get_loss(model, loss_type='all')
print('BEST PARAMS:','\n\nN:', model.N,
      '\nrho:', model.rho,
      '\nsigma: ', model.sigma,
      '\nloss train: ',loss['train'],
      '\nloss validation: ', loss['valid'],
      '\nloss test: ', loss['test'])
get_plot(model)
