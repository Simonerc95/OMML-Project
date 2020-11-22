from functions_12_MSG import *
np.random.seed(1679838)
df = pd.read_csv(r'..\OMML2020_Assignment_1_Dataset.csv')
run_random_search = True

if run_random_search:
    best_model = random_search(model = RBF, iterations=80, df = df, params = params, print_=False, n_jobs=-1)
else:
    best_params = {'N': 16, 'rho': 1e-5, 'sigma': 1.3}
    best_model = RBF(df=df, **best_params)
    best_model.fit()
    print(best_model.fit_time)
best_model.print_loss_params()
get_plot(best_model)