from data_downloader import DataDownloader
import prepare_yield_dataset

if __name__ == "__main__":
    download_Data = DataDownloader()
    download_Data.download_climate_data(params=download_Data.params[1])