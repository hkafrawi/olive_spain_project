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


# Combining Wide Features
wide_features = list(filter(lambda s: s.startswith("Preprocess_") and "_Wide" in s, os.listdir("process_data\\pickle\\climate_polygon_combined")))
wide_features_dataframes = []
for file in wide_features:
    file_path = os.path.join("process_data\\pickle\\climate_polygon_combined",file)
    wide_features_dataframes.append(pd.read_pickle(file_path).copy())

wide_features_combined = pd.concat(wide_features_dataframes,join="outer",axis=1).reset_index()

eub.save_dataframe(wide_features_combined,f"Combined_Wide_Features",location=("process_data\\pickle\\climate_polygon_combined",
                                                                        "process_data\\csv\\climate_polygon_combined"))


# Preprocessing and Merge Wide dataframe
wide_features_combined_v2 = wide_features_combined[(wide_features_combined["Year"]>="1999") &
                                                    (wide_features_combined["Year"]<"2023")].reset_index(drop=True)

wide_features_combined_v2.replace({"Almeria":'Almería',
                 'Ourense':'Orense',
                 'Castellón de la Plana':'Castellón',
                 'A Coruña':'La Coruña',
                 },inplace=True)
wide_features_combined_v2.drop(wide_features_combined_v2.loc[wide_features_combined_v2["Province"].isin(['Vizcaya','Cantabria','Asturias'])].index,inplace=True)

new_columns = [
    (col[0] if col[0] not in ['Year', 'Province'] else '', col[1] if col[0] not in ['Year', 'Province'] else col[0])
    for col in wide_features_combined_v2.columns
]
wide_features_combined_v2.columns.name = None
wide_features_combined_v2.columns = pd.MultiIndex.from_tuples(new_columns)
wide_features_combined_v2.columns = wide_features_combined_v2.columns.droplevel(0)


yield_density = pd.read_pickle("process_data\\pickle\\yield\\Yield_Density_1999_2022_14112024_20_41.pickle")
yield_density.drop(yield_density.loc[yield_density["Province"].isin(['Las Palmas',
                                                        'Barcelona',
                                                        'Gerona',
                                                        'Lérida',
                                                        'Tarragona',
                                                        'Islas Baleares',
                                                        'Santa Cruz de Tenerife'])].index,inplace=True)

working_wide_db = yield_density.merge(wide_features_combined_v2,how='left',left_on=["Year","Province"],right_on=["Year","Province"])

eub.save_dataframe(working_wide_db,f"Wide_Dataframe",location=("data_under_experiment\\pickle\\wide",
                                                        "data_under_experiment\\csv\\wide"))

# Combining Long Features
long_features = list(filter(lambda s: s.startswith("Preprocess_") and "_Long" in s, os.listdir("process_data\\pickle\\climate_polygon_combined")))
long_features_dataframes = []
for file in long_features:
    file_path = os.path.join("process_data\\pickle\\climate_polygon_combined",file)
    long_features_dataframes.append(pd.read_pickle(file_path).copy())

long_features_combined = pd.concat(long_features_dataframes,join="outer",axis=0).reset_index()

eub.save_dataframe(long_features_combined,f"Combined_Long_Features",location=("process_data\\pickle\\climate_polygon_combined",
                                                                        "process_data\\csv\\climate_polygon_combined"))
    
# Preprocessing and Merge Long dataframe
long_features_combined_v2 = long_features_combined[(long_features_combined["Year"]>="1999") &
                                                    (long_features_combined["Year"]<"2023")].reset_index(drop=True)

long_features_combined_v2.replace({"Almeria":'Almería',
                 'Ourense':'Orense',
                 'Castellón de la Plana':'Castellón',
                 'A Coruña':'La Coruña',
                 },inplace=True)

mask = ~long_features_combined_v2["Province"].isin(['Vizcaya','Cantabria','Asturias'])
long_features_combined_v2 = long_features_combined_v2[mask]

working_long_db = yield_density.merge(long_features_combined_v2,how='left',left_on=["Year","Province"],right_on=["Year","Province"])

eub.save_dataframe(working_long_db,f"Long_Dataframe",location=("data_under_experiment\\pickle\\wide",
                                                                "data_under_experiment\\csv\\wide"))    


