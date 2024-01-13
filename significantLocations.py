import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import seaborn as sns
import matplotlib.pyplot as plt
import time
import os

sns.set()

#working with microsoft GeoLife dataset https://www.microsoft.com/en-us/download/details.aspx?id=52367

user = "001"
userdata = 'E:\Projects\Data\Geolife/' + user + '/Trajectory/'

filelist = os.listdir(userdata) 
names = ['latitude','longitude','zero','alt','days','date','time']
df_list = [pd.read_csv(userdata + f,header=6,names=names,index_col=False,) for f in filelist]
df = pd.concat(df_list, ignore_index=True)

df['user_id'] = user
df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])

#df['dow'] = df['timestamp'].dt.day_name
#day of week begins with monday
df['dow'] = df['timestamp'].dt.dayofweek

df.drop(['zero', 'days'], axis=1, inplace=True)

# data is recorded every ~1-5 seconds. Reduce to every minute
df = df.iloc[::12, :]

print(df.head(10))

print('Total #GPS points: ' + str(df.shape[0]))
#df.to_csv('GeoLife_'+user+'_orig_data_resamp.csv')

#Function to calculate the centroid of a cluster from DBSCAN
def get_centroid(cluster):
    cluster_ary = np.asarray(cluster)
    centroid = cluster_ary.mean(axis=0)
    return centroid

#extract coordinates
coords = df.as_matrix(columns=['latitude','longitude'])

#define the number of kilometers in one radian
#use this to convert epsilon from kilometers to radians
kms_per_rad = 6371.0088

#convert espilon to radians for use by haversine distance metric
epsilon = .5/kms_per_rad #1.5=1.5km, 1=1km, 0.5=500m, 0.25=250m, 0.1=100m

start_time = time.time()

#initiate DBSCAN algorithm 
dbsc = DBSCAN(eps=epsilon, min_samples=125, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))

#get the cluster lables
cluster_labels = dbsc.labels_
#get the number of clusters, remove noise (data in cluster -1)
num_clusters = len(set(dbsc.labels_) - set([-1]))

#print number of clusters, compression, and runtime
message = 'Clustered {:,} points down to {:,} clusters, for {:.1f}% compression in {:,.2f} seconds'
print(message.format(len(df), num_clusters, 100*(1 - float(num_clusters) / len(df)), time.time()-start_time))

#turn clusters into a pandas series,where each element is a cluster of points
clusters = pd.Series([coords[cluster_labels==n] for n in range(num_clusters)])
#print(clusters.iloc[0])

# get centroid of each cluster
cluster_centroids = clusters.map(get_centroid)

# unzip the list of centroid points (lat, lon) tuples into separate lat and lon lists
cent_lats, cent_lons = zip(*cluster_centroids)

# from these lats/lons create a new df of one representative point for eac cluster
centroids_df = pd.DataFrame({'longitude':cent_lons, 'latitude':cent_lats})
print(centroids_df.shape)
print(centroids_df.head(10))

#print centroids_df
index_label = ['cluster_label']
header=['latitude','longitude']
centroids_df.to_csv('GeoLife_001_cluster_centroids.csv', index=True, index_label=index_label, header=header)

# Get counts for each cluster
df['cluster_label'] = cluster_labels
cluster_counts = df['cluster_label'].value_counts()
print(cluster_counts.shape)
print(cluster_counts.head(10))

#print cluster_counts
index_label = ['cluster_label']
header = ['frequency']
cluster_counts.to_csv('GeoLife_'+user+'_cluster_counts.csv', index=True, index_label=index_label, header=header)

#output data with cluster assignments, filter out noise (cluster -1)
df_cluster = df[cluster_labels>-1]
cluster_outputs = pd.DataFrame(df_cluster)
print(cluster_outputs.shape)
print(cluster_outputs.head(10))

#print cluster_outputs
cluster_outputs.to_csv('GeoLife_'+user+'_cluster_outputs.csv', index=False, header=True)

#cluster_outputs = pd.DataFrame(cluster_outputs, columns=['user_id','timestamp','latitude','longitude','cluster_label'])

#for data with user_id
#group the data by 'user_id', remove users with less than 5 entries
filtered = cluster_outputs.groupby('user_id').filter(lambda x: len(x['user_id'])>=10)

#sort user locations by time
filtered = filtered.sort_values(by='timestamp')
f = lambda x: [list(x)]
trajs = filtered.groupby('user_id')['cluster_label'].apply(f).reset_index()

print(trajs.shape)
print(trajs.head(10))

#write the output to csv
trajs.to_csv('GeoLife_'+user+'_trajectories_outputs.csv', index=True, index_label='index', header=['user_id','trajectory'])

M = []
def gethour(row):
    t = df[(df['latitude']==row[0]) & (df['longitude']==row[1])]['time'].iloc[0]
    return t[:t.index(':')]
for i in range(num_clusters):
    hours = np.apply_along_axis(gethour, 1, clusters[i]).tolist()
    M.append(list(map(int, hours)))

D = []
def getday(row):
    t = df[(df['latitude']==row[0]) & (df['longitude']==row[1])]['dow'].iloc[0]
    return t
for i in range(num_clusters):
    days = np.apply_along_axis(getday, 1, clusters[i]).tolist()
    D.append(list(map(int, days)))
#print(M)

f, axes = plt.subplots(5, figsize=(12,12),sharex=True,squeeze=True)
sns.distplot(M[0], color="b", ax=axes[0])
sns.distplot(M[1], color="indianred", ax=axes[1])
sns.distplot(M[2], color="b", ax=axes[2])
sns.distplot(M[3], color="indianred", ax=axes[3])
sns.distplot(M[4], color="b", ax=axes[4], axlabel="Time of day")

plt.xticks(np.arange(0, 25, 1.0))
plt.show()
#f.savefig('time-of-day-analysis.png')

#seaborn colors: https://python-graph-gallery.com/100-calling-a-color-with-seaborn/

g, axes = plt.subplots(5, figsize=(12,12),sharex=True,squeeze=True)
sns.distplot(D[0], color="b", ax=axes[0])
sns.distplot(D[1], color="indianred", ax=axes[1])
sns.distplot(D[2], color="b", ax=axes[2])
sns.distplot(D[3], color="indianred", ax=axes[3])
sns.distplot(D[4], color="b", ax=axes[4], axlabel="Day of week")

plt.xticks(np.arange(0, 7, 1.0))
plt.show()
g.savefig('day-of-week-analysis.png')

#seaborn colors: https://python-graph-gallery.com/100-calling-a-color-with-seaborn/

# # Probability Mass Function (PMF) and Normalized cluster count
import math
from collections import Counter

user = trajs['user_id'][0]
cluster = trajs['cluster_label'][0]
#print(user)

counter = Counter(cluster[0])
print(counter)
print("Top 5 clusters: ", counter.most_common(10))

cluster_norm = sum(counter.values(), 0.0)
for key in counter:
    counter[key] /= cluster_norm
print("PMF:",counter.most_common(5))

# # The probability mass function is the function which describes the probability associated with the random variable x. This function is named P(x) or P(x=x) to avoid confusion. P(x=x) corresponds to the probability that the random variable x take the value x (note the different typefaces).

# # *https://hadrienj.github.io/posts/Probability-Mass-and-Density-Functions/*

# # The probability distribution of a discrete random variable is a list of probabilities associated with each of its possible values. It is also sometimes called the probability function or the probability mass function. To have a mathematical sense, suppose a random variable X may take k different values, with the probability that X=xi defined to be P(X=xi)=pi. Then the probabilities pi must satisfy the following:

# # 1: 0 < pi < 1 for each i
# # 2: p1+p2+...+pk=1.

# # *https://www.datacamp.com/community/tutorials/probability-distributions-python*

# # The probability mass function, f(x) = P(X = x), of a discrete random variable X has the following properties:

# # All probabilities are positive: fx(x) ≥ 0.
# # Any event in the distribution (e.g. “scoring between 20 and 30”) has a probability of happening of between 0 and 1 (e.g. 0% and 100%).
# # The sum of all probabilities is 100% (i.e. 1 as a decimal): Σfx(x) = 1.
# # An individual probability is found by adding up the x-values in event A. P(X Ε A) = summation f(x)(xEA)

# # *https://www.quora.com/What-are-the-CDF-PMF-PDF-in-probability*

# # The probability mass function yields the probability of a specific event or probability of a range of events. From this function we can derive the cumulative probability function, F(x)—also called the cumulative distribution function, cumulative mass function, and probability distribution function—defined as that fraction of the total number of possible outcomes X (a random variable), which are less than a specific value x (a number). Thus, the distribution function is the probability that X ≤ x, or

# # *https://www.sciencedirect.com/topics/mathematics/probability-mass-function*

trajs['counter'] = trajs.cluster_label.apply(lambda x: Counter(x[0]))
trajs['top3_clusters'] = trajs.counter.apply(lambda x: x.most_common(3))
def pfm(cntr):
    s = sum(cntr.values())
    for key in cntr:
        cntr[key] /= s
    return cntr.most_common(5)
trajs['cluster_pfm_top5'] = trajs.counter.apply(pfm)
trajs = trajs.drop('counter', 1)
print(trajs.iloc[0])
#trajs.to_csv('pmf_test.csv',index=False)

# # Cosine similarity
# # The cosine similarity between two vectors (or two documents on the Vector Space) is a measure that calculates the cosine of the angle between them. This metric is a measurement of orientation and not magnitude, it can be seen as a comparison between documents on a normalized space because we’re not taking into the consideration only the magnitude of each word count (tf-idf) of each document, but the angle between the documents. What we have to do to build the cosine similarity equation is to solve the equation of the dot product for the

# # And that is it, this is the cosine similarity formula. Cosine Similarity will generate a metric that says how related are two documents by looking at the angle instead of magnitude.

# # *http://blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/*

entity1 = [0,0,0,1,3,2,0,0,1,2]
entity2 = [0,0,1,0,3,1,0,0,2,1]

def dot(A,B): 
    return (sum(a*b for a,b in zip(A,B)))

def cosine_similarity(a,b):
    return dot(a,b) / ( (dot(a,a) **.5) * (dot(b,b) ** .5) )

cosine_similarity(entity1,entity2)







