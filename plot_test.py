from k_centers import *
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score

from sklearn.datasets import make_blobs #remove
import matplotlib.pyplot as plt #remove

data, true_labels = make_blobs(n_samples=50,
                       n_features=2,
                       centers=3,
                       cluster_std=1,
                       center_box=(-10.0, 10.0),
                       shuffle=True,
                       random_state=42)

p = 1
distance_matrix = calc_distance_matrix(data,p)
centers = k_centers(data,3,p)
labels = get_labels(data,centers,distance_matrix)
score = silhouette_score(data, labels, metric='minkowski', p=p)

print(centers)
print(data[centers[0]])
print(data[centers[1]])
print(data[centers[2]])
# plot the 3 clusters
plt.scatter(
    data[labels == centers[0], 0], data[labels == centers[0], 1],
    s=50, c='lightgreen',
    marker='s', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    data[labels == centers[1], 0], data[labels == centers[1], 1],
    s=50, c='orange',
    marker='o', edgecolor='black',
    label='cluster 2'
)

plt.scatter(
    data[labels == centers[2], 0], data[labels == centers[2], 1],
    s=50, c='lightblue',
    marker='v', edgecolor='black',
    label='cluster 3'
)

# plot the centroids
plt.scatter(
    data[centers, 0], data[centers, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids'
)
plt.legend(scatterpoints=1)
plt.grid()
plt.show()

ari = adjusted_rand_score(true_labels, labels)
print(ari)