from functions_21_MSG import *
df = pd.read_csv(r'..\OMML2020_Assignment_1_Dataset.csv')
params = {'N':, 'sigma':, 'rho':}
model = MLP(df=df, **params)
model.fit()
loss = get_loss(model, loss_type='all')
print('BEST PARAMS:','\n\nN:', model['param']['N'],
      '\nrho:', model['param']['rho'],
      '\nsigma: ', model['param']['sigma'],
      '\nloss train: ',loss['train'],
      '\nloss validation: ', loss['valid'],
      '\nloss test: ', loss['test'])
get_plot(model)
