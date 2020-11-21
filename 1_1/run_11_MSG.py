from functions_11_MSG import *

df = pd.read_csv(r'..\OMML2020_Assignment_1_Dataset.csv')
rs = random_search(model = MLP, df = df, params = params, print_=False)
best_model = MLP(df=df, **rs['param'])
best_model.W = rs['weights']['W']
best_model.v = rs['weights']['v']
best_model.b = rs['weights']['b']
loss = get_loss(best_model, loss_type='all')
print('BEST PARAMS:','\n\nN:', best_model.N,
      '\nrho:', best_model.rho,
      '\nsigma: ', best_model.sigma,
      '\nloss train: ',loss['train'],
      '\nloss validation: ', loss['valid'],
      '\nloss test: ', loss['test'])
get_plot(best_model)
