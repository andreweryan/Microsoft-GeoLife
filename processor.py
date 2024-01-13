import os
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import warnings

warnings.filterwarnings("ignore")


# data dowload: https://download.microsoft.com/download/F/4/8/F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip

def process_data(user_id, path, df_list):
    df = pd.read_csv(path, skiprows=6, names=['latitude', 'longitude', 'zero', 'altitude', 'days_since', 'date_str', 'time_str'])
    df = df[df['latitude'].between(-90, 90) & df['longitude'].between(-180, 180)]
    df.drop(['zero', 'days_since'], axis=1, inplace=True)
    df['timestamp'] = pd.to_datetime(df['date_str'] + ' ' + df['time_str'])
    df['user_id'] = user_id
    # data is recorded every ~1-5 seconds. Reduce to every minute
    df_resampled = df.resample('1T', on='timestamp').first().reset_index(drop=True)
    gdf = gpd.GeoDataFrame(df_resampled, geometry=gpd.points_from_xy(df_resampled.longitude, df_resampled.latitude), crs="EPSG:4326")
    df_list.append(gdf)
    return None

data_dir = r'C:\Users\andrr\Desktop\geolife\Data'

files = [os.path.join(foldername, filename) for foldername, _, filenames in os.walk(data_dir) for filename in filenames if filename.endswith('.plt')]

file_map = list()

df_list = list()

with tqdm(total=len(files), desc="Processing Data", unit="file") as progress_bar:
    for file in files:
        path, folder = os.path.split(os.path.dirname(file))
        p, user_id = os.path.split(path)
        process_data(user_id, file, df_list)
        progress_bar.update(1)

data = pd.concat(df_list, axis=0, ignore_index=True)
print(data.head())
print(f"Number of records in the dataset: {len(data)}")

# for file in files:
#     path, folder = os.path.split(os.path.dirname(file))
#     p, user_id = os.path.split(path)
#     file_map.append([user_id, file])

# with tqdm(total=len(file_map), desc="Processing Data", unit="file") as progress_bar:
#     with ThreadPoolExecutor(max_workers=None) as executor:
#         futures = {executor.submit(process_data, file[0], file[1], df_list): file for file in file_map}
#         for _ in as_completed(futures):
#             progress_bar.update(1)

gdf = gpd.GeoDataFrame(pd.concat(df_list, axis=0, ignore_index=True))
print(gdf.head())
print(f"Number of records in the dataset: {len(gdf)}")

out_path = os.path.join(os.path.dirname(data_dir), 'geolife_points.parquet')

gdf.to_parquet(out_path)