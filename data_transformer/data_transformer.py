import warnings
from experiment_utilities import ExperimentUtilityBox as eub
import pandas as pd
import xarray as xr
import os
from datetime import datetime

class DataTransformer():
    def __init__(self):
        pass

    @staticmethod
    def prepare_yield_dataset(yield_folder_path):

        # Suppress the specific warning from openpyxl
        warnings.filterwarnings("ignore", category=UserWarning, message="Workbook contains no default style, apply openpyxl's default")

        dataframes = [eub.clean_files(str(yield_folder_path+f"/{file}")) for file in os.listdir(yield_folder_path)]
        df = pd.concat(dataframes,join="outer",axis=1)

        df = df.reset_index()
        dfv2 = df.drop(0,axis=0
        ).rename(columns={'Unnamed: 1':"Province"}
        ).set_index("Province").stack(
        ).reset_index(
        ).rename(columns={"level_1":"Year",0:"Yield_Density"})

        return dfv2
    
    @staticmethod
    def prepare_climate_dataset(file_path:str) -> pd.DataFrame:
        """
        Prepares a climate dataset from a NetCDF (.nc) file.

        This function processes a NetCDF file obtained from Copernicus, extracts data for the 
        Spanish region, handles missing values, and generates a weekly time aggregation. 
        Specifically, it:
        - Opens the NetCDF file and extracts the relevant data array based on the variable inferred from the filename.
        - Focuses on a subset of the dataset corresponding to the Spanish region.
        - Drops any rows with missing (NaN) values.
        - Adds a new column representing the week of the year (ISO week format).
        - Groups the data by week, latitude, and longitude, calculating the mean for each group.

        Parameters:
            file_path (str): The file path to the NetCDF (.nc) file.

        Returns:
            pd.DataFrame: A DataFrame containing the weekly aggregated climate data for the Spanish region 
                        with columns ['Week', 'latitude', 'longitude', '<variable>'].
        """
        ds = xr.open_dataset(file_path)
        variable = os.path.basename(file_path)[:2]
        da = ds[variable]

        spanish_region = da[0:-1,100:200,150:250]
        df = spanish_region.to_dataframe().reset_index()
        df = df.dropna()
        df["Week"] = df["time"].dt.strftime("%Y-%U")

        return df.groupby(["Week","latitude","longitude"]).agg({variable:'mean'}).reset_index()