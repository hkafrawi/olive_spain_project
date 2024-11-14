
import cdsapi
import configparser
from datetime import datetime
import os

class DataDownloader:
    def __init__(self,logger=None):
        self.datasets = ['insitu-gridded-observations-europe']
        self.params = [{
                        "product_type": "ensemble_mean",
                        "variable": [
                            "mean_temperature",
                            "minimum_temperature",
                            "maximum_temperature",
                            "precipitation_amount"
                        ],
                        "grid_resolution": "0_25deg",
                        "period": "2011_2024",
                        "version": ["30_0e"]
                        },{
                        "product_type": "ensemble_mean",
                        "variable": [
                            "mean_temperature",
                            "minimum_temperature",
                            "maximum_temperature",
                            "precipitation_amount"
                        ],
                        "grid_resolution": "0_25deg",
                        "period": "1995_2010",
                        "version": ["30_0e"]
                    }]
        # Acquiring API Keys
        config = configparser.ConfigParser()
        config.read("config.ini")

        self.id = config.get("API","ID")
        self.api_key = config.get("API","KEY")
        self.url = config.get("API","URL")
        self.key = {self.id:self.api_key}
        
        self.folder_path = "raw_data/climate"
        os.makedirs(self.folder_path, exist_ok=True)

    
    def download_climate_data(self, datasets = None,
                 params = None,
                 file_name= None):
       
       if not datasets:
           datasets = self.datasets[0]
       if not params:
           params = self.params[0]
       if not file_name:
           file_name=f"{datetime.now().strftime('%d%m%Y %H_%M')}.zip"
           file_path = os.path.join(self.folder_path, file_name)
       else:
           file_path = os.path.join(self.folder_path, file_name)
           

       c = cdsapi.Client(url=str(self.url),key=str(self.api_key))
       print(datasets)
       c.retrieve(
            name=datasets,
            request = params,
            target = file_path)
       return file_path