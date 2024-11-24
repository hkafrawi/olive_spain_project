import os
import pickle
import pandas as pd
import numpy as np
from experiment_utilities import ExperimentUtilityBox as eub
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV 

class Experiment:
    def __init__(self, data, target_column, dataset_name, date_column=None):
        self.data = data
        self.target_column = target_column
        self.date_column = date_column
        self.dataset_name = dataset_name
        self.train_data = None
        self.test_data = None

        self.pipeline = None
        self.search = None
        self.best_params = None
        self.experiment_name = None

    def train_test_split(self, test_size=0.2, random_state=42):
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        self.train_data, self.test_data = train_test_split(
            pd.concat([X, y], axis=1),
            test_size=test_size,
            random_state=random_state
        )
        

    def date_split(self, split_date, drop_date=True):
        """
        Splits the data into train and test sets based on a specified date.

        Args:
            split_date (str or datetime): The date to split the data.
            drop_date (bool, optional): Whether to drop the date column from train and test datasets. Defaults to True.
        """
        if not self.date_column:
            raise ValueError("Date Column is not defined.")

        # Split the data
        self.train_data = self.data[self.data[self.date_column] < split_date]
        self.test_data = self.data[self.data[self.date_column] >= split_date]

        # Drop the date column if specified
        if drop_date:
            self.train_data = self.train_data.drop(columns=[self.date_column])
            self.test_data = self.test_data.drop(columns=[self.date_column])

    def setup_pipeline(self,steps):
        self.pipeline = Pipeline(steps)

        self.experiment_name = "_".join(type(step[1]).__name__ for step in steps)

    def run_search(self, param_grid, iterations=1, cv=5,n_jobs = 1, random_state=42):
        if not self.pipeline:
            raise ValueError("Pipeline is not defined. Call setup_pipeline first.")
        if type(iterations) != int:
            raise TypeError("Iterations has to be an int more than 1. Default: 1")
        
        self.hash_param_grid = eub.compute_hash(str(param_grid))
        
        # Attempt to load the existing experiment if the experiment name is set
        if not self._load_existing_experiment():
        
            X_train = self.train_data.drop(columns=[self.target_column])
            y_train = self.train_data[self.target_column]

            if iterations == 1:
                for hyperparamteres in param_grid.values():
                    iterations*=len(hyperparamteres)
            if iterations < 100:
                self.search = GridSearchCV(
                    self.pipeline, param_grid, cv=cv, scoring='r2', n_jobs=n_jobs,
                    verbose=2
                )
            else:
                iteration_num = int(np.ceil(0.05*iterations))
                self.search = RandomizedSearchCV(
                    self.pipeline, param_grid, cv=cv, n_iter=iteration_num, 
                    scoring='r2', n_jobs=n_jobs, random_state=random_state,
                    verbose=2
                )
            

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

    def evaluate(self, y_test, y_pred, metrics):
        results = {
        "Model": type(self.search.best_estimator_.named_steps['model']).__name__,
        "Best_Params": self.best_params,
        "Dataset_Name": self.dataset_name,
        "Experiment_Name": self.experiment_name,
        "Observations": len(self.train_data)
        }

        for name, metric in metrics.items():
            results[name] = metric(y_test, y_pred)
        return results
    
    def save_experiment(self):
        if self.experiment_name is None:
            raise ValueError("Experiment name is not set. Please set up the pipeline first.")
        
        experiment_path = f"experiments/{self.experiment_name}_{self.dataset_name}_{self.hash_param_grid}_experiment.pkl"
        os.makedirs(os.path.dirname(experiment_path), exist_ok=True)

        # Save the entire Experiment instance to a file
        with open(experiment_path, 'wb') as exp_file:
            pickle.dump(self, exp_file)
        print(f"Experiment saved to {experiment_path}")

    def _load_existing_experiment(self):
        if self.experiment_name is None:
            return  # We can't load without an experiment name
        experiment_path = f"experiments/{self.experiment_name}_{self.dataset_name}_{self.hash_param_grid}_experiment.pkl"
        if os.path.exists(experiment_path):
            # Load the entire experiment
            with open(experiment_path, 'rb') as exp_file:
                loaded_experiment = pickle.load(exp_file)
                self.__dict__.update(loaded_experiment.__dict__)  # Update this instance with loaded data
            print(f"Experiment loaded from {experiment_path}")
            return True
        else:
            print(f"No existing experiment found for Experiment '{self.experiment_name}' and Dataset '{self.dataset_name}'. Starting a new experiment.")
            return False
    
    @staticmethod
    def cluster_data(data, target_column, dataset_name):
        """Clusters the data using KMeans with the optimum number of clusters determined by the elbow method.
        
        Args:
            data (pd.DataFrame): The input dataset.
            target_column (str): The name of the target column to exclude from clustering.
            dataset_name (str): The base name of the dataset.

        Returns:
            list[dict]: A list of dictionaries containing cluster-specific datasets.
        """

        def optimum_cluster(features):
            """Utility function to compute the optimum number of clusters using the elbow method."""
            wcss = []
            for i in range(1, 11):
                kmeans = KMeans(n_clusters=i, random_state=42)
                kmeans.fit(features)
                wcss.append(kmeans.inertia_)

            # Calculate the "elbow" point using the second derivative
            wcss_diff = np.diff(wcss)  # First derivative
            wcss_diff2 = np.diff(wcss_diff)  # Second derivative

            # Find the index where the second derivative changes the most (highest positive value)
            elbow_index = np.argmax(wcss_diff2) + 2  # +2 accounts for diff shifts
            return elbow_index

        # Exclude the target column for clustering
        features = data.drop(columns=[target_column])

        # Determine the optimum number of clusters
        n_clusters = optimum_cluster(features)

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        data['Cluster'] = kmeans.fit_predict(features)

        # Create a list of dictionaries for each cluster
        clustered_data = []
        for cluster_id in range(n_clusters):
            cluster_data = data[data['Cluster'] == cluster_id].copy()
            cluster_data.drop(columns=['Cluster'], inplace=True)
            clustered_data.append({
                'Dataset_name': f"{dataset_name}_Cluster_{cluster_id}",
                'Data': cluster_data
            })

        return clustered_data
