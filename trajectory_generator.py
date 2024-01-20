import gc
import momepy
import osmnx as ox
import numpy as np
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import LineString

from haversine.haversine import haversine_distance  # custom

import warnings

warnings.filterwarnings("ignore")

gc.enable()


def make_trajectory(points_df, time_col="timestamp"):
    """Make trajectory."""
    points = points_df.sort_values(time_col)

    coords = [(p.x, p.y) for p in points.geometry]

    trajectory = LineString(coords)

    return trajectory


gdf = gpd.read_parquet(r"C:\Projects\data\geolife\geolife_points_5min.parquet")

# this code should be in processor during polars transforms
gdf.sort_values(by=["user_id", "timestamp"], inplace=True)
gdf["end_time"] = gdf.groupby("user_id")["timestamp"].shift(-1)
gdf["end_latitude"] = gdf.groupby("user_id")["latitude"].shift(-1)
gdf["end_longitude"] = gdf.groupby("user_id")["longitude"].shift(-1)

gdf["distance_kilometers"] = gdf.apply(
    lambda row: haversine_distance(
        row["latitude"],
        row["longitude"],
        row["end_latitude"],
        row["end_longitude"],
        unit="kilometers",
    ),
    axis=1,
)
gdf = gdf[gdf["distance_kilometers"] >= 0.5].copy()
gdf["time_minutes"] = (gdf["end_time"] - gdf["timestamp"]).dt.total_seconds() / 60
gdf["time_hours"] = (gdf["end_time"] - gdf["timestamp"]).dt.total_seconds() / 3600
gdf = gdf[gdf["time_minutes"] >= 5].copy()
gdf["velocity_kmh"] = gdf["distance_kilometers"] / gdf["time"]

gdf_filtered = gdf[(gdf["velocity_kmh"] >= 10) & (gdf["velocity_kmh"] <= 80)].copy()
gdf_filtered.head()

# user_ids = np.unique(gdf['user_id'].values)

# gdf['date'] = gdf['timestamp'].dt.date
# trajectories_list = list()

# with tqdm(total=len(user_ids), desc="Processing Data", unit="user") as progress_bar:
#     for user_id in user_ids:
#         user_df = gdf[(gdf.user_id == user_id)]
#         user_trajectory = make_trajectory(user_df)
#         trajectories_list.append([user_id, user_trajectory])
#         progress_bar.update(1)

# trajs_df = pd.DataFrame(trajectories_list)
# trajs_df.rename({0: "user_id", 1:"geometry"}, axis=1, inplace=True)
# trajs_gdf = gpd.GeoDataFrame(trajs_df, geometry='geometry')
# trajs_gdf = trajs_gdf.set_crs('EPSG:4326')
# trajs_gdf.head()
# trajs_gdf.to_parquet(r'C:\Users\andrr\Desktop\geolife\geolife_trajectories.parquet')

# graph stuff
# G = momepy.gdf_to_nx(gdf, approach="primal")
# nodes, edges = momepy.nx_to_gdf(G)
# nodes['x'], nodes['y'] = nodes.geometry.x, nodes.geometry.y
# edges['u'] = edges['node_start']
# edges['v'] = edges['node_end']
# edges['k'] = 0
# edges.set_index(['u','v','k'], inplace=True)
