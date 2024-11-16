from itertools import chain
import warnings

from shapely import Point
from experiment_utilities import ExperimentUtilityBox as eub
import pandas as pd
import xarray as xr
import os
import geopandas as gpd
from datetime import datetime

class DataTransformer():
    def __init__(self):
        pass

    @staticmethod
    def prepare_yield_dataset(yield_folder_path:str) -> pd.DataFrame:

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
    def prepare_climate_dataset(file_path:str, x=115,y=30) -> pd.DataFrame:
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
            x (int): the starting latitude point that will extend to 50
            y (int): the starting longitude point that will extend to 50

        Returns:
            pd.DataFrame: A DataFrame containing the weekly aggregated climate data for the Spanish region 
                        with columns ['Week', 'latitude', 'longitude', '<variable>'].
        """
        ds = xr.open_dataset(file_path)
        variable = os.path.basename(file_path)[:2]
        da = ds[variable]

        spanish_region = da[:,y:y+50,x:x+50]
        df = spanish_region.to_dataframe().reset_index()
        df = df.dropna()
        df["Week"] = df["time"].dt.strftime("%Y-%U")

        return df.groupby(["Week","latitude","longitude"]).agg({variable:'mean'}).reset_index()
    
    @staticmethod
    def prepare_polygon_dataset(filepath:str) -> pd.DataFrame:
        """
        Prepares the polygon dataset from a .shp file.

        This function processes a .shp file by dissolving the dataset by province,
        then computing the centroid for each region as well as its latitude and longitude
        points.

        Parameters:
            file_path (str): The file path to the .shp file.

        Returns:
            pd.DataFrame       
        """
        df = gpd.read_file(filepath)
        spain_df = df.dissolve(by="DS_PROVINC")

        spain_df["centroid"] = spain_df["geometry"].apply(lambda x:x.centroid)
        spain_df["lat"] = spain_df["centroid"].apply(lambda point:point.y)
        spain_df["lon"] = spain_df["centroid"].apply(lambda point:point.x)

        return spain_df
    
    @staticmethod
    def get_points_in_spain(spain_pickle_path:str, points_pickle_path:str) -> pd.DataFrame:
        """
        Matches points from a global dataset with Spanish regions based on a pickle file
        and returns a subset DataFrame using spatial indexing for faster performance.

        Parameters:
            spain_pickle_path (str): Path to the pickle file containing Spanish region polygons as a GeoDataFrame.
            points_pickle_path (str): Path to the pickle file containing points data as a Dataframe.

        Returns:
            pd.DataFrame: A DataFrame containing week, variable, and matched region (province) name.
        """
        # Load the Spanish region data from the pickle file
        spain_gdf = pd.read_pickle(spain_pickle_path)

        # Load the points data from the pickle file and sets variable name
        points_df = pd.read_pickle(points_pickle_path)
        variable_name = points_df.columns[-1]

        # Create a GeoDataFrame for points
        points_gdf = gpd.GeoDataFrame(
            points_df,
            geometry=[Point(x,y) for x,y in zip(points_df['longitude'], points_df['latitude'])],
            crs="EPSG:4258"
        )

        # Ensure that both GeoDataFrames have the same CRS
        if spain_gdf.crs != points_gdf.crs:
            points_gdf = points_gdf.to_crs(spain_gdf.crs)

        # Extract unique points
        unique_points_gdf = points_gdf.drop_duplicates(subset=['longitude', 'latitude'])

        print(f"Unique Points_gdf Shape: {unique_points_gdf.shape}")
        print(f"spain_gdf Shape: {spain_gdf.shape}")

        # Initialize a dictionary to store matched results for unique points
        matched_points = {}

        # Loop through each unique point and check if it is within any of the polygons in Spain
        for index, point_row in unique_points_gdf.iterrows():
            point = point_row['geometry']
            matching_provinces = spain_gdf[spain_gdf['geometry'].contains(point)]

            if not matching_provinces.empty:
                # Use the first matching province (if multiple matches exist)
                province_name = matching_provinces.iloc[0].name
                matched_points[(point_row['longitude'], point_row['latitude'])] = province_name

        # Create an empty list to store the results
        matched_data = []

        # Map the results back to the original points_gdf
        for index, point_row in points_gdf.iterrows():
            lon_lat_tuple = (point_row['longitude'], point_row['latitude'])
            province_name = matched_points.get(lon_lat_tuple, None)

            if province_name:
                matched_data.append({
                    'Week': point_row['Week'],
                    variable_name: point_row[variable_name],
                    'Province': province_name
                })

        # Create a DataFrame from the matched results
        result_df = pd.DataFrame(matched_data)

        return result_df

    @staticmethod
    def prepare_wide_dataset(dfs, variable):
        combined_dataframe = pd.concat(dfs)
        combined_dataframe["Year"] = combined_dataframe["Week"].apply(lambda x: x.split("-")[0])
        combined_dataframe["Week"] = combined_dataframe["Week"].apply(lambda x: "{}_Week_{}".format(variable, x.split("-")[1]))

        final_dataframe = combined_dataframe.pivot_table(index=["Year","Province"],
                                                         columns="Week")
        
        return final_dataframe
    
    @staticmethod
    def prepare_long_dataset(dfs, variable):
        combined_dataframe = pd.concat(dfs)
        combined_dataframe["Year"] = combined_dataframe["Week"].apply(lambda x: x.split("-")[0])
        combined_dataframe["Week"] = combined_dataframe["Week"].apply(lambda x: x.split("-")[1])

        final_dataframe = combined_dataframe.melt(id_vars=["Year","Week","Province"],
                                                value_vars=variable)
        
        return final_dataframe
        
