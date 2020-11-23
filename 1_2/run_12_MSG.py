from functions_12_MSG import *
np.random.seed(1679838)
df = pd.read_csv(r'..\OMML2020_Assignment_1_Dataset.csv')
run_random_search = False

if run_random_search:
    best_model = random_search(model = RBF, iterations=80, df=df, params=params, print_=False, n_jobs=-1)
else:
    best_params = {'N': 29, 'rho': 1e-5, 'sigma': 0.8}
    best_model = RBF(df=df, **best_params)
    best_model.fit(print_=False, maxiter=800)

best_model.print_loss_param()

get_plot(best_model)