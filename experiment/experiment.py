import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV 

class Experiment:
    def __init__(self, data, target_column, date_column=None):
        self.data = data
        self.target_column = target_column
        self.date_column = date_column
        self.train_data = None
        self.test_data = None

        self.pipeline = None
        self.search = None
        self.best_param = None

    def train_test_split(self, test_size=0.2, random_state=42):
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        self.train_data, self.test_data = train_test_split(
            pd.concat([X, y], axis=1),
            test_size=test_size,
            random_state=random_state
        )
        

    def date_split(self, split_date):
        if not self.date_column:
            raise ValueError("Date Column is not defined.")
        self.train_data = self.data[self.data[self.date_column] < split_date]
        self.test_data = self.data[self.data[self.date_column] >= split_date]

    def setup_pipeline(self,steps):
        self.pipeline = Pipeline(steps)

    def run_search(self, param_grid, search_type="grid", cv=5, n_iter=10, random_state=42):
        if not self.pipeline:
            raise ValueError("Pipeline is not defined. Call setup_pipeline first.")
        
        X_train = self.train_data.drop(columns=[self.target_column])
        y_train = self.train_data[self.target_column]

        # Choose search method
        if search_type == "grid":
            self.search = GridSearchCV(
                self.pipeline, param_grid, cv=cv, scoring='r2', n_jobs=-1,
                verbose=2
            )
        elif search_type == "random":
            self.search = RandomizedSearchCV(
                self.pipeline, param_grid, cv=cv, n_iter=n_iter, 
                scoring='r2', n_jobs=-1, random_state=random_state,
                verbose=2
            )
        else:
            raise ValueError("Invalid search_type. Choose 'grid' or 'random'.")

        # Perform search
        self.search.fit(X_train, y_train)

    def run_pipeline(self):
        if not self.search:
            raise ValueError("Search is not run. Call run_search first.")

        X_test = self.test_data.drop(columns=[self.target_column])
        y_test = self.test_data[self.target_column]

        # Use the best model for predictions
        self.best_params = self.search.best_params_

        best_pipeline = self.search.best_estimator_
        y_pred = best_pipeline.predict(X_test)

        return y_test, y_pred

    def evaluate(self):
        pass
