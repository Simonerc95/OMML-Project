from functions_3_MSG import *

df = pd.read_csv(r'../OMML2020_Assignment_1_Dataset.csv')
params = {'N': 29, 'rho': 1e-05, 'sigma': 0.8}
model = RBF(df=df, **params)
model.two_block(num_iter=20)
loss = get_loss(model, loss_type='all')

print('BEST PARAMS:','\n\nN:', model.N,
      '\nrho:', model.rho,
      '\nsigma: ', model.sigma,
      '\nloss train: ',loss['train'],
      '\nloss validation: ', loss['valid'],
      '\nloss test: ', loss['test'])
get_plot(model)
