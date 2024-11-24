from functools import wraps
import hashlib
from itertools import chain
import pickle
import os
from datetime import datetime
import traceback
import pandas as pd
import numpy as np
import warnings
import zipfile

class ExperimentUtilityBox():
    """
    A utility class for aiding in the experimentation process, providing
    methods for logging, saving and loading objects, and managing directories.

    This class is designed to simplify common tasks, such as logging experiment 
    details, saving models and other objects, and handling file operations.
    """
    def __init__(self):
        pass

    @staticmethod
    def save_dataframe(pandas_object: pd.DataFrame, db_name:str, location:tuple = None):
        """
        Saves a Pandas DataFrame as both a pickle and a CSV file.

        This method saves a given Pandas DataFrame to two locations:
        1. A pickle file for fast loading.
        2. A CSV file for easy data inspection and compatibility.

        The method creates the necessary directories if they do not exist. If a custom location is not provided,
        the DataFrame is saved to default directories (`/saved_dataframes/pickle` and `/saved_dataframes/csv`).

        Args:
            pandas_object (pd.DataFrame): The Pandas DataFrame to save.
            db_name (str): The base name for the saved files. The current date and time will be appended.
            location (tuple, optional): A tuple containing custom paths for saving the pickle and CSV files.
                - `location[0]`: Path for saving the pickle file.
                - `location[1]`: Path for saving the CSV file.
                If not provided, defaults are used.

        Raises:
            FileNotFoundError: If the provided custom paths do not exist and cannot be created.
            Exception: If there is an issue during the saving process (e.g., file permissions, invalid DataFrame).

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
            >>> ExperimentUtilities.save_dataframe(df, "my_dataframe")
            >>> ExperimentUtilities.save_dataframe(df, "my_dataframe", ("/custom/pickle_path", "/custom/csv_path"))
        """

        if not location:
            pickle_folder_path = "saved_dataframes\pickel"
            csv_folder_path = "saved_dataframes\csv"
        else:
            pickle_folder_path = location[0]
            csv_folder_path = location[1]

        os.makedirs(pickle_folder_path,exist_ok=True)
        os.makedirs(csv_folder_path, exist_ok=True)
        db_name = f"{db_name}_{datetime.now().strftime('%d%m%Y_%H_%M')}"

        pk_folder_path = os.path.join(pickle_folder_path,db_name)
        print(pk_folder_path)
        c_folder_path = os.path.join(csv_folder_path,db_name)

        file_path = open(f"{pk_folder_path}.pickle","ab")
        pickle.dump(pandas_object,file_path)
        pandas_object.to_csv(f"{c_folder_path}.csv")

        return pk_folder_path, c_folder_path
    
    @staticmethod
    def clean_files(filename):
        try:
            df = pd.read_excel(filename)
            df2 = df.iloc[:-2,1:3]
            df2.rename(columns={df2.columns[-1]:filename.split("/")[-1].split(".")[0].split("/")[-1]},inplace=True)
            df2.set_index(df2.columns[0],inplace=True)
            df2.drop(["Total",np.nan],axis=0,inplace=True)
            return df2
        except Exception as e:
            print(e)
            pass
    
    @staticmethod
    def unzip_files(zip_path:str, location:str) -> None:
        try:
            # Open the ZIP file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract all files
                zip_ref.extractall(location)
                print(f"{zip_path} extracted successfully to '{location}'")
        except Exception as e:
            print(e)
        else:
            os.remove(zip_path)
            print(f"{zip_path} has been deleted")

    @staticmethod
    def log_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            print(f"Starting {func.__name__} with args: {args} and kwargs: {kwargs}")
            
            try:
                result = func(*args, **kwargs)  # Call the original function with instance
                return result
            except Exception as e:
                print(f"Error in {func.__name__}: {e}")
                print(traceback.print_exc())
            finally:
                end_time = datetime.now()
                elapsed_time = end_time - start_time
                total_seconds = elapsed_time.total_seconds()
                hours, remainder = divmod(total_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                print(f"Finished {func.__name__}")
                print(f"{func.__name__} took {int(hours)} hrs {int(minutes)} mins {int(seconds)} seconds to compute")
        
        return wrapper
    
    @staticmethod
    def compute_hash(hashable_string):
        return hashlib.sha256(hashable_string.encode()).hexdigest()
