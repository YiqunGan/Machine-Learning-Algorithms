import numpy as np
x= np.array([[1,2],[2,3],[4,7],[2,5],[6,12],[6,9]])
center_points = np.array([[2,2],[6,5]])

N= 6
D =2
n_cluster =2

J = 10 ** 10
count = 0

x_ones = np.ones((N, n_cluster, D))
x_m = x_ones * np.expand_dims(x, axis=1)
center_ones = np.ones((N, n_cluster, D))
while count < 20:
    count += 1
    center_m = center_ones * center_points
    dist_m = np.sum((x_m - center_m) ** 2, axis=2)
    Gamma = np.zeros((N, n_cluster))
    min_idx = np.argmin(dist_m, axis=1)
    for i in range(N):
        Gamma[i][min_idx[i]] = 1
    J1 = np.sum(Gamma * dist_m)
    print(J1)
    if np.abs(J - J1) < 0.0000001:
        break;
    J = J1
    center_points = np.sum(x_m * np.expand_dims(Gamma, axis=2), axis=0) / np.expand_dims(np.sum(Gamma, axis=0), axis=1)
    print(center_points)
max_iter = count
center_m = center_ones * center_points
dist_m = np.sum((x_m - center_m) ** 2, axis=2)
min_idx = np.argmin(dist_m, axis=1)
y = min_idx
centroids = center_points
print(max_iter)



        kmeans = KMeans(self.n_cluster,self.max_iter,self.e,self.generator)
        centroids, k_index, max_iter = kmeans.fit(x, centroid_func)
        array_dist = []
        for i in range(self.n_cluster):
            array_dist.append({})
        for m,n in zip(y,k_index):
            if m not in array_dist[n].keys():
                array_dist[n][m]=1
            else:
                array_dist[n][m]+=1
        centroid_labels = np.zeros(self.n_cluster)
        for j in range(self.n_cluster):
            centroid_labels[j]= max(array_dist[j],key = array_dist[j].get)

kmeans = KMeans(self.n_cluster, self.max_iter, self.e, self.generator)
centroids, k_index, max_iter = kmeans.fit(x, centroid_func)
array_dist = []
for i in range(self.n_cluster):
    array_dist.append({})
for m, n in zip(y, k_index):
    if m not in array_dist[n].keys():
        array_dist[n][m] = 1
    else:
        array_dist[n][m] += 1
centroid_labels = np.zeros(self.n_cluster)
for j in range(self.n_cluster):
    centroid_labels[j] = max(array_dist[j], key=array_dist[j].get)
