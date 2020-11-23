from functions_22_MSG import *

df = pd.read_csv(r'../OMML2020_Assignment_1_Dataset.csv')
params = {'N': 29, 'rho': 1e-05, 'sigma': 0.8}
model = RBF(df=df, **params)
model.LLSP(num_iter=5000)
model.print_loss_params()
get_plot(model)
