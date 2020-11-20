from functions_12_MSG import *

df = pd.read_csv(r'..\OMML2020_Assignment_1_Dataset.csv')
rs = random_search(model = RBF, df = df, params = params, print_=True)
model = RBF(df=df, **rs['param'])
model.fit()
loss = get_loss(model, loss_type='all')
print('BEST PARAMS:','\n\nN:', rs['param']['N'],
      '\nrho:', rs['param']['rho'],
      '\nsigma: ', rs['param']['sigma'],
      '\nloss train: ',loss['train'],
      '\nloss validation: ', loss['valid'],
      '\nloss test: ', loss['test'])
get_plot(model)