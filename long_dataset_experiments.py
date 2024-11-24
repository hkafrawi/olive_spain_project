import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.svm import SVR
from experiment_utilities import ExperimentUtilityBox as eub
from experiment import Experiment
import xgboost as xgb

data = pd.read_pickle("data_under_experiment\\pickle\wide\Long_Dataframe_22112024_18_05.pickle")
data["Week"] = data["Week"].astype(int)
data["Year"] = data["Year"].astype(int)
data = data[(data["Week"] > 0)&(data["Week"] <=30 )]
data = pd.get_dummies(data, columns=['variable'], drop_first=True)
data = pd.get_dummies(data, columns=['Province'], drop_first=True)
data = data.dropna(axis=0)
data = data.drop("index",axis=1)

def MAPE(testing_data, prediected_data):
    return mean_absolute_percentage_error(testing_data, prediected_data)

def MAXP(testing_data, prediected_data):
    percentage_errors = np.abs((testing_data - prediected_data)/testing_data) * 100
    return np.max(percentage_errors)

def r2_metric(testing_data, prediected_data):
    return r2_score(testing_data, prediected_data)

metrics = {
    'MAPE': MAPE,
    'MAXP': MAXP,
    'r2': r2_metric
}

models_long_dataset = {
        LinearRegression():{
                                "model__fit_intercept": [True],
                                "model__copy_X": [True],
                                "model__n_jobs": [None]
                            },
        Ridge():{'model__alpha': [0.01, 0.1, 1, 10, 100]},
        Lasso():{'model__alpha': [0.01, 0.1, 1, 10, 100]},
        ElasticNet():{'model__alpha': [0.01, 0.1, 1, 10, 100]},
        RandomForestRegressor():{
                                'model__n_estimators': [50, 100, 200],  # Number of trees in the forest
                                'model__max_depth': [None, 10, 20, 30],  # Maximum depth of the trees
                                'model__min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
                                'model__min_samples_leaf': [1, 2, 4]     # Minimum number of samples required to be at a leaf node
        },
        xgb.XGBRegressor(objective='reg:squarederror', device='cuda'):{
                                'model__n_estimators': [100, 200, 300],
                                'model__learning_rate': [0.01, 0.05, 0.1],
                                'model__max_depth': [3, 5, 7],
                                'model__subsample': [0.7, 0.8, 1.0],
                                'model__colsample_bytree': [0.7, 0.8, 1.0],
                                'model__gamma': [0, 0.1, 0.2],
                                'model__reg_alpha': [0, 0.1, 0.5],
                                'model__reg_lambda': [1, 1.5, 2],
}}

@eub.log_decorator
def run_scalar_models(data=data):
    results = []
    for model, params in models_long_dataset.items():
        exp = Experiment(data, target_column='Yield_Density', dataset_name="Long_Dataset_Random_Split", date_column='Year')
        exp.train_test_split()
        exp.setup_pipeline([
                            ('scaler', StandardScaler()),
                            ('model', model)
                        ])
        param_grid = params
        exp.run_search(param_grid, iterations=1, cv=5,n_jobs=10)
        y_test, y_pred = exp.run_pipeline()
        result = exp.evaluate(y_test, y_pred, metrics)
        results.append(result)
        exp.save_experiment()

    for model, params in models_long_dataset.items():
        exp = Experiment(data, target_column='Yield_Density', dataset_name="Long_Dataset_Date_Split", date_column='Year')
        exp.date_split(split_date='2016')
        exp.setup_pipeline([
                            ('scaler', StandardScaler()),
                            ('model', model)
                        ])
        param_grid = params
        exp.run_search(param_grid, iterations=1, cv=5,n_jobs=10)
        y_test, y_pred = exp.run_pipeline()
        result = exp.evaluate(y_test, y_pred, metrics)
        results.append(result)
        exp.save_experiment()

    return results

@eub.log_decorator
def run_normal_models(data=data):
    results = []
    for model, params in models_long_dataset.items():
        exp = Experiment(data, target_column='Yield_Density', dataset_name="Long_Dataset", date_column='Year')
        exp.train_test_split()
        exp.setup_pipeline([
                            ('model', model)
                        ])
        param_grid = params
        exp.run_search(param_grid, iterations=1, cv=5,n_jobs=10)
        y_test, y_pred = exp.run_pipeline()
        result = exp.evaluate(y_test, y_pred, metrics)
        results.append(result)
        exp.save_experiment()

    for model, params in models_long_dataset.items():
        exp = Experiment(data, target_column='Yield_Density', dataset_name="Long_Dataset_Date_Split", date_column='Year')
        exp.date_split(split_date='2016')
        exp.setup_pipeline([
                            ('model', model)
                        ])
        param_grid = params
        exp.run_search(param_grid, iterations=1, cv=5,n_jobs=10)
        y_test, y_pred = exp.run_pipeline()
        result = exp.evaluate(y_test, y_pred, metrics)
        results.append(result)
        exp.save_experiment()

    return results

@eub.log_decorator
def run_cluster_models(data=data):
    clustered_data = Experiment.cluster_data(data=data,target_column="Yield_Density",dataset_name="Long_Dataset")
    results = []
    for item in clustered_data:
        data = item['Data']
        dataset_name= item["Dataset_name"]

        for model, params in models_long_dataset.items():
            exp = Experiment(data, target_column='Yield_Density', dataset_name=dataset_name, date_column='Year')
            exp.train_test_split()
            exp.setup_pipeline([
                                ('model', model)
                            ])
            param_grid = params
            exp.run_search(param_grid, iterations=1, cv=5,n_jobs=10)
            y_test, y_pred = exp.run_pipeline()
            result = exp.evaluate(y_test, y_pred, metrics)
            results.append(result)
            exp.save_experiment()

        for model, params in models_long_dataset.items():
            exp = Experiment(data, target_column='Yield_Density', dataset_name=f"{dataset_name}_Date_Split", date_column='Year')
            exp.date_split(split_date='2016')
            exp.setup_pipeline([
                                ('model', model)
                            ])
            param_grid = params
            exp.run_search(param_grid, iterations=1, cv=5,n_jobs=10)
            y_test, y_pred = exp.run_pipeline()
            result = exp.evaluate(y_test, y_pred, metrics)
            results.append(result)
            exp.save_experiment()
            
    return results


if __name__ == "__main__":
    pass