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

# # Obtain frequency of edges to set directed edge weights
# c = Counter(g.edges())  # Contains frequencies of each directed edge.

# minLineWidth = 0.5 # Set minimum line width to scale edge widths

# for u, v, d in g.edges(data=True):
#     d['weight'] = c[u, v]*minLineWidth
# edges,weights = zip(*nx.get_edge_attributes(g,'weight').items())
# weightlist = (list(g.edges(data=True)))

# # Save edge weights to file
# edgeweight = pd.DataFrame(weightlist)
# #print edgeweight
# header=['cluster_from','cluster_to','weight']
# edgeweight.to_csv('G:\Projects\Data\dmv_tweets_march_10d_merged_edges_weights.csv', index=False, header=header)
# Plotting with matplotlib
# nx.draw_networkx_nodes(g,pos=dict_pos,with_labels=True,node_size=25,node_color='black',alpha=.25)
# nx.draw_networkx_edges(g,pos=dict_pos,arrows=False,width=[d['weight'] for u,v, d in g.edges(data=True)], edge_color=colorList)
# plt.show()
