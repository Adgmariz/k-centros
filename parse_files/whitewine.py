import numpy as np
import pandas as pd
from k_centers import *
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import time

filepath = 'datasets/wine+quality/winequality-white.csv'
output_file = 'whitewine.csv'
k = 11
numbers_per_line = 12 #including the class

def calc_radius_kmeans(data, centers):
    # Calculate the Minkowski distance from each data point to each center
    distances = pairwise_distances(data, centers, metric='minkowski', p=1)
    # Find the minimum distance from each data point to the centers
    min_distances = np.min(distances, axis=1)
    # The radius is the maximum of these minimum distances
    radius = np.max(min_distances)
    return radius

df = pd.read_csv(filepath, sep=';',skiprows=1)
data = df.drop(df.columns[-1], axis=1).values
true_labels = df[df.columns[-1]].values

with open(output_file, 'w') as f:
    for p in range(1,4):
        start_time = time.time()
        distance_matrix = calc_distance_matrix(data,p)
        end_time = time.time()
        processing_time = end_time - start_time

        start_time = time.time()
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        s_score = silhouette_score(data, kmeans.labels_, metric='minkowski', p=p)
        ari = adjusted_rand_score(true_labels, kmeans.labels_)
        radius = radius = calc_radius_kmeans(data, kmeans.cluster_centers_)
        f.write(f"{p}")
        f.write(f",{radius}")
        f.write(f",{s_score}")
        f.write(f",{ari}")
        # Stop the timer
        end_time = time.time()
        # Calculate the time difference and print it
        processing_time = end_time - start_time
        f.write(f",{processing_time}\n")

        for i in range(30):

            start_time = time.time()
            centers = k_centers(data,k,p)
            labels = get_labels(data,centers,distance_matrix)
            radius = calc_radius(data, centers, distance_matrix)
            s_score = silhouette_score(data, labels, metric='minkowski', p=p)
            ari = adjusted_rand_score(true_labels, labels)
            f.write(f"{p}")
            f.write(f",{radius}")
            f.write(f",{s_score}")
            f.write(f",{ari}")
            end_time = time.time()
            processing_time = end_time - start_time
            f.write(f",{processing_time}\n")