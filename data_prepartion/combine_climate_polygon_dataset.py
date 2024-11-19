from data_transformer import DataTransformer as dt
from experiment_utilities import ExperimentUtilityBox as eub
import os

polygon_dataset = "process_data\pickle\Spain_Polygon_14112024_22_51.pickle"
files_directory = "process_data\\pickle\\climate"

preliminary_files = [os.path.join(files_directory,file) for file in os.listdir(files_directory) if file.startswith("PreliminaryProcess")]


for file in preliminary_files:

    print(f"Running through {file}")
    df = dt.get_points_in_spain(polygon_dataset,file)

    variable_name = os.path.basename(file).split("_")[1]
    date_range = os.path.basename(file).split("_")[6]
    file_name = f"{variable_name}_{date_range}"

    eub.save_dataframe(df,file_name,location=("process_data\\pickle\\climate_polygon_combined",
                                            "process_data\\csv\\climate_polygon_combined"))