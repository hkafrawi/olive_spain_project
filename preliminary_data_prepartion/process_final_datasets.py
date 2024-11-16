import os
import pandas as pd
import numpy as np
from data_transformer import DataTransformer as dt
from experiment_utilities import ExperimentUtilityBox as eub

datasets = {"rr":[],
            "tg":[],
            "tn":[],
            "tx":[]}

directory = "process_data\pickle\climate_polygon_combined"

for file in os.listdir(directory):
    file_path = os.path.join(directory,file)
    variable = file[:2]
    dataframe = pd.read_pickle(file_path)
    datasets[variable].append(dataframe)

for variable, dataframes in datasets.items():
    df_wide = dt.prepare_wide_dataset(dataframes,variable)
    eub.save_dataframe(df_wide,f"Preprocess_{variable}_Wide",location=("process_data\\pickle\\climate_polygon_combined",
                                                                        "process_data\\csv\\climate_polygon_combined"))
    df_long = dt.prepare_long_dataset(dataframes,variable)
    eub.save_dataframe(df_long,f"Preprocess_{variable}_Long",location=("process_data\\pickle\\climate_polygon_combined",
                                                                        "process_data\\csv\\climate_polygon_combined"))
    
    


