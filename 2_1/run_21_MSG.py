from functions_21_MSG import *
np.random.seed(1679838)
df = pd.read_csv(r'..\OMML2020_Assignment_1_Dataset.csv')
params = {'N': 28, 'rho': 1e-4, 'sigma': 0.7}
model = MLP(df=df, **params)
model.extreme_learning(num_iter=6000)
model.print_loss_params()
get_plot(model)
