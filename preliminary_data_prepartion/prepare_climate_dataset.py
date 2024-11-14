from experiment_utilities import ExperimentUtilityBox as eub
from data_transformer import DataTransformer as dt
import os

folder_path = "raw_data\climate"
climate_datasets = os.listdir(folder_path)
print(climate_datasets)

zip_files = [file for file in climate_datasets if file.endswith(".zip")]
for file in zip_files:
    file_path = os.path.join(folder_path,file)
    eub.unzip_files(file_path,folder_path)

nc_files = [file for file in climate_datasets if file.endswith(".nc")]

for file in nc_files:
    print(f"Processing {file}")
    file_path = os.path.join(folder_path,file)
    df = dt.prepare_climate_dataset(file_path)
    eub.save_dataframe(df,f"PreliminaryProcess_{file}",location=("process_data\\pickle",
                                                          "process_data\\csv"))


