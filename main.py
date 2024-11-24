from data_downloader import DataDownloader 
from data_prepartion import combine_climate_polygon_dataset
from data_prepartion import process_final_datasets
from experiment_utilities import ExperimentUtilityBox as eub
import pandas as pd
import data_prepartion as dp
from data_transformer import DataTransformer as dt
import wide_dataset_experiments, long_dataset_experiments
from datetime import datetime
import traceback




if __name__ == "__main__":
    # download_Data = DataDownloader()
    # download_Data.download_climate_data(params=download_Data.params[1])

    # combine_climate_polygon_dataset.main()
    # process_final_datasets.main()

    # eub.move_directory_with_timestamp("process_data","backup\\backup")

    logger = eub.generate_log_file()
    eub.stream_to_logger(logger=logger)
    
    results = []
    for experiment, methods in [
    (wide_dataset_experiments, ['run_normal_models', 'run_scalar_models', 'run_cluster_models']),
    (long_dataset_experiments, ['run_normal_models', 'run_scalar_models', 'run_cluster_models'])
]:
        for method in methods:
            try:
                results.extend(getattr(experiment, method)())  # Dynamically call the method
            except Exception as e:
                
                print(f"Error occurred while executing {experiment}.{method}:")
                print(traceback.format_exc())  # Print the full traceback but continue

            finally:
                # Save results
                df = pd.DataFrame(results)
                eub.save_dataframe(df, f"Evaluation_Wide_Dataset_{datetime.now().strftime('%d_%m_%Y %H_%M')}")