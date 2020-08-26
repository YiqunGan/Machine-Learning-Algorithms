import numpy as np
y=[0,1,2,1,1,1,2,2,0,0]
n_cluster = 4
membership = [0,1,1,2,3,2,1,2,1,3]
voting = []

array_dist = []
for i in range(n_cluster):
    array_dist.append({})
for m, n in zip(y, membership):
    if m not in array_dist[n].keys():
        array_dist[n][m] = 1
    else:
        array_dist[n][m] += 1
centroid_labels = np.zeros(n_cluster)
for j in range(n_cluster):
    centroid_labels[j] = max(array_dist[j], key=array_dist[j].get)


print(centroid_labels)