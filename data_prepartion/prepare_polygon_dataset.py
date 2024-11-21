from experiment_utilities import ExperimentUtilityBox as eub
from data_transformer import DataTransformer as dt
import os

@eub.log_decorator
def main():
    polygon_file_name = "ComarcasAgrarias.shp"
    polygon_folder_path = "raw_data\spain_polgon"

    file_path = os.path.join(polygon_folder_path,polygon_file_name)
    df = dt.prepare_polygon_dataset(file_path)

    pickle_folder_path = "process_data\\pickle"
    csv_folder_path = "process_data\\csv"
    pickle_saved_df, _ = eub.save_dataframe(df,"Spain_Polygon",location=(pickle_folder_path,csv_folder_path))

    return pickle_saved_df

if __name__ == "__main__":
    main()