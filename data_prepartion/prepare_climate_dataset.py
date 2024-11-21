from experiment_utilities import ExperimentUtilityBox as eub
from data_transformer import DataTransformer as dt
from copy import copy
import os

@eub.log_decorator
def main():
    folder_path = "raw_data\climate"
    climate_datasets = os.listdir(folder_path)

    try:
        zip_files = [file for file in climate_datasets if file.endswith(".zip")]
        for file in zip_files:
            file_path = os.path.join(folder_path,file)
            eub.unzip_files(file_path,folder_path)
    except Exception as e:
        print(e)

    nc_files = [file for file in climate_datasets if file.endswith(".nc")]

    climate_files = []
    for file in nc_files:
        print(f"Processing {file}")
        file_path = os.path.join(folder_path,file)
        df = dt.prepare_climate_dataset(file_path)
        climate_file, _ = eub.save_dataframe(df,f"PreliminaryProcess_{file}",location=("process_data\\pickle\\climate",
                                                            "process_data\\csv\\climate"))
        climate_files.append(copy(climate_file))

    return climate_files
        
if __name__ == "__main__":
    main()
