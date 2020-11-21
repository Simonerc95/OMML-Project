from functions_12_MSG import *
np.random.seed(1679838)
df = pd.read_csv(r'..\OMML2020_Assignment_1_Dataset.csv')
rs = random_search(model = RBF, df = df, params = params, print_=False)
best_model = RBF(df, **rs['param'])
best_model.C = rs['weights']['C']
best_model.v = rs['weights']['v']
loss = get_loss(best_model, loss_type='all')
print('BEST PARAMS:','\n\nN:', best_model.N,
      '\nrho:', best_model.rho,
      '\nsigma: ', best_model.sigma,
      '\nloss train: ',loss['train'],
      '\nloss validation: ', loss['valid'],
      '\nloss test: ', loss['test'])
get_plot(best_model)