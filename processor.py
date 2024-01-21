import os
import gc
from pathlib import Path
import polars as pl
from glob import glob
from datetime import datetime
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import LineString
from concurrent.futures import ProcessPoolExecutor, as_completed

from haversine.haversine import (
    haversine_distance,
)  # https://github.com/andreweryan/haversine

import warnings

warnings.filterwarnings("ignore")

gc.enable()

# data dowload: https://download.microsoft.com/download/F/4/8/F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip


def extract_user_id(data_path):
    return int(Path(data_path).parts[-3])


def process_data(path, resample="1m"):
    df = pl.read_csv(
        path,
        skip_rows=6,
        columns=[0, 1, 5, 6],
        new_columns=["latitude", "longitude", "date_str", "time_str"],
        dtypes=[pl.Float64, pl.Float64, pl.String, pl.String],
        rechunk=True,
    )

    df = df.with_columns(
        pl.concat_str(
            [pl.col("date_str"), pl.col("time_str")],
            separator=" ",
        ).alias("timestamp_str")
    )

    df = df.with_columns(
        pl.col("timestamp_str").str.to_datetime().cast(pl.Datetime).alias("timestamp")
    )

    df = df.sort(pl.col("timestamp"), descending=False)

    df = (
        df.set_sorted("timestamp")
        .group_by_dynamic(
            "timestamp", every=resample
        )  # data is recorded every ~1-5 seconds.
        .agg(pl.col(pl.Float64).mean())
    )

    df = df.drop(["date_str", "time_str"])

    user_id = extract_user_id(path)
    df = df.with_columns(pl.lit(user_id).alias("user_id"))
    df = df.sort(["user_id", "timestamp"])
    df = df.with_columns(
        pl.col(["timestamp", "latitude", "longitude"])
        .shift(-1)
        .over("user_id")
        .name.prefix("end_")
    )

    df = df.drop_nulls()

    df = df.with_columns(
        pl.struct(["latitude", "longitude", "end_latitude", "end_longitude"])
        .map_elements(
            lambda x: haversine_distance(
                x["latitude"],
                x["longitude"],
                x["end_latitude"],
                x["end_longitude"],
                "kilometers",
            ),
            return_dtype=pl.Float64,
        )
        .alias("distance_kilometers")
    )

    df = df.with_columns(
        pl.col(["timestamp", "end_timestamp"]),
        delta=((pl.col("end_timestamp").sub(pl.col("timestamp")) / 1000000) / 3600),
    )

    df = df.rename({"delta": "time_delta_hr"})

    df = df.with_columns(
        (pl.col("distance_kilometers") / pl.col("time_delta_hr")).alias("speed_kmh")
    )

    if resample == "30s":
        time_lim = 0.5 / 60
    elif resample == "90s":
        time_lim = 1.5 / 60
    elif resample == "1m":
        time_lim = 1 / 60
    elif resample == "5m":
        time_lim = 5 / 60
    else:
        raise ValueError(
            "Selected resample was not implemented. Select from: 30s, 90s, 1min, 5min"
        )

    df = df.filter(
        (pl.col("time_delta_hr") == time_lim)
        & (pl.col("distance_kilometers") > 0.2)
        & (pl.col("distance_kilometers") < 2)
        & (pl.col("speed_kmh") <= 80)
    )

    return df


def trip_to_line(row):
    return LineString(
        [
            [row["longitude"], row["latitude"]],
            [row["end_longitude"], row["end_latitude"]],
        ]
    )


if __name__ == "__main__":
    start = datetime.now()

    data_dir = r"C:\Projects\data\geolife\Data"

    files = glob(r"C:\Projects\data\geolife\Data\*\*\*.plt")

    df_list = list()

    with tqdm(total=len(files), desc="Processing Data", unit="file") as progress_bar:
        with ProcessPoolExecutor(max_workers=None) as executor:
            futures = {
                executor.submit(process_data, file, resample="30s"): file
                for file in files
            }
            for future in as_completed(futures):
                result = future.result()
                df_list.append(result)
                progress_bar.update(1)

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

    # process trip points to lines/trajectories
    gdf["line_geom"] = gdf.apply(lambda row: trip_to_line(row), axis=1)
    gdf.drop("geometry", axis=1, inplace=True)
    gdf.rename(columns={"line_geom": "geometry"}, inplace=True)
    gdf.set_geometry("geometry", inplace=True)
    gdf.set_crs("EPSG:4326", inplace=True)

    out_path = os.path.join(os.path.dirname(data_dir), "geolife_lines.parquet")
    gdf.to_parquet(out_path)

    end = datetime.now()
    print(f"Total processing time: {end - start}")
