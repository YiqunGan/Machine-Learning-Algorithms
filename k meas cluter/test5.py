import numpy as np
x= np.array([[1,2,3],[2,3,5],[4,2,1],[2,2,3],[6,5,5],[6,7,1],[11,2,3],[2,7,9]])
center_points = np.array([[2,2,3],[1,2,1],[6,5,3],[3,3,1]])
'''
N, M, C = x.shape
data = x.reshape(N * M, C)
code_vectors = center_points
z = data - np.expand_dims(code_vectors, axis=1)
y=np.sum(((data - np.expand_dims(code_vectors, axis=1)) ** 2), axis=2)
r = np.argmin(np.sum(((data - np.expand_dims(code_vectors, axis=1)) ** 2), axis=2), axis=0)
new_im = code_vectors[r].reshape(N, M, C)
print(new_im)

N, M, D = x.shape
code_vectors = center_points
N1, D = code_vectors.shape
pixel = x.reshape(N * M, D)
'''
y = []
for i in range(8):
    y.append(i)
centers = [1,2,3,4]
centers= np.array(centers)
centroids = x[np.ix_(centers)]
y= np.array(y)
y[3]=2
y[4]=0
y[5]=2
y[6]=1
y[7]=3
print(centroids)
#print(x[y==2])
for k in range(4):
    if len(x[y == k]) != 0:
        center_points[k] = np.mean(x[y == k], axis=0,dtype=np.float64)
#print(center_points)



