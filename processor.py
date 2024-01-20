import os
import gc
from pathlib import Path
import numpy as np
import polars as pl
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import geopandas as gpd
from concurrent.futures import ThreadPoolExecutor, as_completed

from haversine.haversine import (
    haversine_distance,
)  # https://github.com/andreweryan/haversine

import warnings

warnings.filterwarnings("ignore")

gc.enable()

# data dowload: https://download.microsoft.com/download/F/4/8/F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip

start = datetime.now()


def extract_user_id(data_path):
    return int(Path(data_path).parts[-3])


def process_data(path, df_list, resample="5m"):
    df = pl.read_csv(
        path,
        skip_rows=6,
        columns=[0, 1, 3, 5, 6],
        new_columns=["latitude", "longitude", "altitude", "date_str", "time_str"],
        dtypes=[pl.Float64, pl.Float64, pl.Float64, pl.String, pl.String],
    )

    df = df.with_columns(
        pl.concat_str(
            [pl.col("date_str"), pl.col("time_str")],
            separator=" ",
        ).alias("timestamp_str")
    )

    df = df.with_columns(
        pl.col("timestamp_str").str.to_datetime().cast(pl.Datetime).alias("start_time")
    )

    """data is recorded every ~1-5 seconds. Reduce/downsample to every 10s"""
    df = (
        df.set_sorted("start_time")
        .group_by_dynamic("start_time", every=resample)
        .agg(pl.col(pl.Float64).mean())
    )

    df = df.drop(["altitude", "date_str", "time_str"])

    df = df.with_columns(pl.col("start_time").shift(-1).alias("end_time"))

    df = df.with_columns(pl.col("latitude").shift(-1).alias("end_latitude"))

    df = df.with_columns(pl.col("longitude").shift(-1).alias("end_longitude"))

    df = df.drop_nulls()

    df = df.with_columns(
        pl.struct(["latitude", "longitude", "end_latitude", "end_longitude"])
        .map_elements(
            lambda x: haversine_distance(
                x["latitude"],
                x["longitude"],
                x["end_latitude"],
                x["end_longitude"],
                "meters",
            ),
            return_dtype=pl.Float64,
        )
        .alias("distance_meters")
    )

    df = df.with_columns(
        pl.col(["start_time", "end_time"]),
        delta=pl.col("end_time").sub(pl.col("start_time")),
    )

    user_id = extract_user_id(path)

    df = df.with_columns(pl.lit(user_id).alias("user_id"))

    df_list.append(df)

    df = None

    return None


data_dir = r"C:\Projects\data\geolife\Data"

files = [
    os.path.join(foldername, filename)
    for foldername, _, filenames in os.walk(data_dir)
    for filename in filenames
    if filename.endswith(".plt")
]

df_list = list()

with tqdm(total=len(files), desc="Processing Data", unit="file") as progress_bar:
    for file in files:
        process_data(file, df_list, resample="1m")
        progress_bar.update(1)

# with tqdm(total=len(files), desc="Processing Data", unit="file") as progress_bar:
#     with ThreadPoolExecutor(max_workers=None) as executor:
#         futures = {executor.submit(process_data, file, df_list): file for file in files}
#         for _ in as_completed(futures):
#             progress_bar.update(1)

df = pl.concat(df_list)
gdf = gpd.GeoDataFrame(
    df.to_pandas(),
    geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
    crs="EPSG:4326",
)
df = None
print(f"Number of records in the dataset: {len(gdf)}")
print(gdf.head())

out_path = os.path.join(os.path.dirname(data_dir), "geolife_points.parquet")

gdf.to_parquet(out_path)

end = datetime.now()
print(f"Total processing time: {end - start}")
