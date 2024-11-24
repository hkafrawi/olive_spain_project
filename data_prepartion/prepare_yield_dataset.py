from experiment_utilities import ExperimentUtilityBox as eub
from data_transformer import DataTransformer as dt
import os

@eub.log_decorator
def main():
    yield_folder_path = "raw_data\yield"
    yield_dataframe = dt.prepare_yield_dataset(yield_folder_path)

    print("Yield Dataset Sample:")
    print(yield_dataframe.head())

    print("Saving Final Yield Dataset...")

    pickle_folder_path = "process_data\pickle\yield"
    csv_folder_path = "process_data\csv\yield"

    pickle_saved_dataframe, _ = eub.save_dataframe(yield_dataframe,"Yield_Density_1999_2022",location=(pickle_folder_path,csv_folder_path))

    return pickle_saved_dataframe

if __name__ == "__main__":
    main()