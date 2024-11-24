import os
import pandas as pd
import numpy as np
from data_prepartion import prepare_yield_dataset
from data_transformer import DataTransformer as dt
from experiment_utilities import ExperimentUtilityBox as eub


def main():
    print(f"Running {__name__}")
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

    # Combine Features together
    wide_features_combined = dt.combine_features("Wide","process_data\\pickle\\climate_polygon_combined")
    long_features_combined = dt.combine_features("Long","process_data\\pickle\\climate_polygon_combined")

    # Prepare Yield Dataset
    yield_file_location = f"{prepare_yield_dataset.main()}.pickle"
    yield_density = pd.read_pickle(yield_file_location)
    yield_density.drop(yield_density.loc[yield_density["Province"].isin(['Las Palmas',
                                                            'Barcelona',
                                                            'Gerona',
                                                            'Lérida',
                                                            'Tarragona',
                                                            'Islas Baleares',
                                                            'Santa Cruz de Tenerife'])].index,inplace=True)


    # Finalizing Final Wide Dataframe
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

    working_wide_db = yield_density.merge(wide_features_combined_v2,how='left',left_on=["Year","Province"],right_on=["Year","Province"])

    eub.save_dataframe(working_wide_db,f"Wide_Dataframe",location=("data_under_experiment\\pickle\\wide",
                                                            "data_under_experiment\\csv\\wide"))

        
    # Finalizing Final Long Dataframe

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

if __name__ == "__main__":
    main()