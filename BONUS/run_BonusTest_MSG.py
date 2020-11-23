from functions_BonusTest_MSG import *
np.random.seed(1679838)
df = pd.read_csv(r'..\OMML2020_Assignment_1_Dataset.csv')
path = r'.\DataPointsBonusTest.csv'  # please, put here path of the bonus test dataset
test_dataset = pd.read_csv(path)


with open('best_model_bonus.pkl', 'rb') as pkl:
    best_model = pickle.load(pkl)


best_model.print_params()
best_model.predict_and_rmse_base(df)
best_model.predict_and_rmse_test(test_dataset)
get_plot(best_model, df)
