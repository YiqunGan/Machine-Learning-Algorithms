J = 10 ** 10
        iteration = 0
        center_points = x[np.ix_(self.centers)]
        while iteration < self.max_iter:
            iteration += 1
            y = []
            J_F = 0
            #center_numerator = np.zeros((self.n_cluster, D))
            #center_denominater = np.zeros((self.n_cluster))
            r_nk = np.zeros((N,self.n_cluster))
            x_curr = np.ones((N,self.n_cluster,D))*np.expand_dims(x,axis=1)
            for i in range(N):
                d_min = 10 ** 5
                d_index = 0
                for j in range(self.n_cluster):
                    d_curr = np.sum((center_points[j] - x[i]) ** 2)
                    if d_curr < d_min:
                        d_index = j
                        d_min = d_curr
                J_F = J_F + d_min
                y.append(d_index)
                r_nk[i][d_index]=1
                #center_numerator[d_index] = center_numerator[d_index] + x[i]
                #center_denominater[d_index] = center_denominater[d_index] + 1
            if np.abs(J - J_F) <= self.e:
                break;
            J = J_F
            # print(center_denominater)
            #center_points = center_numerator / center_denominater[:, None]
            center_points = np.sum(x_curr * np.expand_dims(r_nk, axis=2), axis=0) / np.expand_dims(np.sum(r_nk, axis=0),
                                                                                                 axis=1)
        self.max_iter = iteration
        centroids = center_points
        y= np.array(y)
        return centroids, y, self.max_iter

J = 10**5
        iteration = 0
        center_points = x[np.ix_(self.centers)]
        while iteration < self.max_iter:
            iteration += 1
            y = []
            J_F = 0
            # center_numerator = np.zeros((self.n_cluster, D))
            # center_denominater = np.zeros((self.n_cluster))
            r_nk = np.zeros((N, self.n_cluster))
            x_curr = np.ones((N, self.n_cluster, D)) * np.expand_dims(x, axis=1)
            for i in range(N):
                d_min = 10 ** 5
                d_index = 0
                for j in range(self.n_cluster):
                    d_curr = np.sum((center_points[j] - x[i]) ** 2)
                    if d_curr < d_min:
                        d_index = j
                        d_min = d_curr
                J_F = J_F + d_min
                y.append(d_index)
                r_nk[i][d_index] = 1
                # center_numerator[d_index] = center_numerator[d_index] + x[i]
                # center_denominater[d_index] = center_denominater[d_index] + 1
            if np.abs(J - J_F) < self.e:
                break;
            J = J_F
            # print(center_denominater)
            # center_points = center_numerator / center_denominater[:, None]
            center_points = np.sum(x_curr * np.expand_dims(r_nk, axis=2), axis=0) / np.expand_dims(np.sum(r_nk, axis=0),
                                                                                                   axis=1)
        self.max_iter = iteration
        centroids = center_points
        y = np.array(y)
        return centroids, y, self.max_iter



N, M, D = image.shape
    N1,D = code_vectors.shape
    pixel = image.reshape(N*M,D)

    for i in range(N):
        d_min = np.sum((code_vectors[0] - pixel[0]) ** 2)
        d_index = 0
        for j in range(N1):
            d_curr = np.sum((code_vectors[j] - pixel[i]) ** 2)
            if d_curr < d_min:
                d_index = j
                d_min = d_curr
        pixel[i]=code_vectors[d_index]
    new_im = pixel.reshape(N,M,3)
