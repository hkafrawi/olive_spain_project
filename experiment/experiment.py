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

    def setup_pipeline(self):
        pass

    def run_search(self):
        pass

    def run_pipeline(self):
        pass

    def evaluate(self):
        pass
