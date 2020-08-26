import numpy as np
x= np.array([[1,2,5],[2,3,7],[4,2,5],[2,2,1],[6,5,2],[6,7,1],[11,2,5],[7,9,10]])
center_points = np.array([[2,2,4],[6,5,5],[8,6,10]])

N= 8
D =3
n_cluster =3
'''
#print(x[0]+x[1])
J = 10 ** 10
iteration = 0

while iteration < 20:
    iteration += 1
    y = []
    J_F = 0
    center_numerator = np.zeros((n_cluster, D))
    center_denominater = np.zeros((n_cluster))
    for i in range(N):
        d_min = 10 ** 10
        d_index = 0
        for j in range(n_cluster):
            d_curr = np.sum((center_points[j] - x[i]) ** 2)
            if d_curr < d_min:
                d_index = j
                d_min = d_curr
        J_F = J_F + d_min
        y.append(d_index)
        center_numerator[d_index] = center_numerator[d_index] + x[i]
        center_denominater[d_index] = center_denominater[d_index] + 1
    if np.abs(J - J_F) < 0.00000001:
        break;
    J = J_F
    # print(center_denominater)
    center_points = center_numerator / center_denominater[:, None]
max_iter = iteration
centroids = center_points
y = np.array(y)

'''

J = 10^10
count = 0
x_ones = np.ones((N, n_cluster, D))
x_m = x_ones * np.expand_dims(x, axis = 1)
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
    if np.abs(J - J1) < 0.00000001:
        break;
    J = J1
    center_points = np.sum(x_m * np.expand_dims(Gamma, axis=2), axis=0) / np.expand_dims(np.sum(Gamma, axis=0), axis=1)
    print(np.expand_dims(np.sum(Gamma, axis=0), axis=1))
max_iter = count
center_m = center_ones * center_points
dist_m = np.sum((x_m - center_m) ** 2, axis=2)
min_idx = np.argmin(dist_m, axis=1)
y = min_idx
centroids = center_points
'''

'''
'''
def distortion_value(x, y, centroids):
    distortion_measure = np.sum([np.sum((x[y == i] - centroids[i]) ** 2) for i in range(n_cluster)])
    return distortion_measure


y = np.zeros(N)
iter_time = 0
centroids=center_points
init_distortion = distortion_value(x, y, centroids)
centroids_n = center_points
distance = np.zeros((n_cluster, N, D))
while iter_time < 20:
    distance = np.sum((x - np.expand_dims(centroids, axis=1)) ** 2, axis=2)
    y = np.argmin(distance, axis=0)
    curr_distortion = distortion_value(x, y, centroids)
    if abs(init_distortion - curr_distortion) <= 0.00000001:
        break
    init_distortion = curr_distortion
    for k in range(n_cluster):
        centroids_n[k] = np.mean(x[y == k], axis=0)
    centroids_n[np.where(np.isnan(centroids_n))] = centroids[np.where(np.isnan(centroids_n))]
    centroids = centroids_n
    iter_time += 1
max_iter = iter_time
print(curr_distortion)
# raise Exception('Implement fit function in KMeans class')

# DO NOT CHANGE CODE BELOW THIS LINE
#return centroids, y, self.max_iter
'''
'''
J = 10 ** 10
iteration = 0

while iteration < 20:
    iteration += 1
    y = []
    J_F = 0
    # center_numerator = np.zeros((self.n_cluster, D))
    # center_denominater = np.zeros((self.n_cluster))
    for i in range(N):
        d_min = np.sum((center_points[0] - x[i]) ** 2)
        d_index = 0
        for j in range(n_cluster):
            d_curr = np.sum((center_points[j] - x[i]) ** 2)
            if d_curr < d_min:
                d_index = j
                d_min = d_curr
        J_F = J_F + d_min
        y.append(d_index)
        # center_numerator[d_index] = center_numerator[d_index] + x[i]
        # center_denominater[d_index] = center_denominater[d_index] + 1
    y = np.array(y)

    for k in range(n_cluster):
        if len(x[y == k]) != 0:
            center_points[k] = np.mean(x[y == k], axis=0)
    if np.abs(J - J_F) <= 0.00001:
        break
    J = J_F
            # print(center_denominater)
            #new_center = center_numerator / center_denominater[:, None]
            #new_center[np.where(np.isnan(new_center))] = center_points[np.where(np.isnan(new_center))]
            #center_points = new_center
max_iter = iteration
centroids = center_points
'''



print(centroids,max_iter)




