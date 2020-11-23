from functions_11_MSG import *
np.random.seed(1679838)
df = pd.read_csv(r'..\OMML2020_Assignment_1_Dataset.csv')
run_random_search = False

if run_random_search:
    best_model = random_search(model=MLP, df=df, params=params, print_=False, n_jobs=20)
else:
    best_params = dict(N=28, rho=1e-4, sigma=0.7)
    best_model = MLP(df=df, **best_params)
    best_model.fit(print_=False, maxiter=1500)
best_model.print_loss_param()
get_plot(best_model)
