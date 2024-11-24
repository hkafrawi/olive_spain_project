from data_prepartion import prepare_climate_dataset, prepare_polygon_dataset
from data_transformer import DataTransformer as dt
from experiment_utilities import ExperimentUtilityBox as eub
import os

@eub.log_decorator
def main():
    polygon_dataset = prepare_polygon_dataset.main()
    polygon_dataset = f"{polygon_dataset}.pickle"
    files_directory = prepare_climate_dataset.main()
    preliminary_files = [f"{file}.pickle" for file in files_directory]


    for file in preliminary_files:

        print(f"Running through {file}")
        df = dt.get_points_in_spain(polygon_dataset,file)

        variable_name = os.path.basename(file).split("_")[1]
        date_range = os.path.basename(file).split("_")[6]
        file_name = f"{variable_name}_{date_range}"

        eub.save_dataframe(df,file_name,location=("process_data\\pickle\\climate_polygon_combined",
                                                "process_data\\csv\\climate_polygon_combined"))
        
if __name__ == "__main__":
    main()