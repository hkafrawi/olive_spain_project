from experiment_utilities import ExperimentUtilityBox as eub
import os

folder_path = "raw_data\climate"
climate_datasets = os.listdir(folder_path)
print(climate_datasets)

for file in climate_datasets:
    file_path = os.path.join(folder_path,file)
    eub.unzip_files(file_path,folder_path)

