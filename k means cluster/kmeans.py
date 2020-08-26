import numpy as np



def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data-  numpy array of points
    :param generator: random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being index to a sample
             which is chosen as centroid.
    '''
    # TODO:
    # implement the Kmeans++ algorithm of how to choose the centers according to the lecture and notebook
    # Choose 1st center randomly and use Euclidean distance to calculate other centers.
    centers = []
    center0 = generator.randint(n)
    centers.append(center0)
    for k in range(n_cluster-1):
        d_square = []

        for i in range(len(x)):
            d_center = []
            for j in centers:
                d_curr = np.sum((x[j]-x[i])**2)
                d_center.append(d_curr)
            d_square.append(min(d_center))
        centers.append(np.argmax(d_square/sum(d_square)))







    #raise Exception(
             #'Implement get_k_means_plus_plus_center_indices function in Kmeans.py')

    

    # DO NOT CHANGE CODE BELOW THIS LINE

    print("[+] returning center for [{}, {}] points: {}".format(n, len(x), centers))
    return centers



def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)




class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a length (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates a Int)
            Note: Number of iterations is the number of time you update the assignment
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        #raise Exception(
            # 'Implement fit function in KMeans class')
        
        # DO NOT CHANGE CODE BELOW THIS LINE
        J = 10 ** 10
        iteration = 0
        center_points = x[np.ix_(self.centers)]
        ones_temp = np.ones((N,self.n_cluster,D))
        x_temp = ones_temp * np.expand_dims(x,axis =1)
        while iteration < self.max_iter:
            iteration += 1
            center_temp = ones_temp* center_points
            distance = np.sum((x_temp-center_temp)**2,axis=2)
            r_nk = np.zeros((N, self.n_cluster))
            y = np.argmin(distance,axis=1)
            for i in range(N):
                r_nk[i][y[i]]=1
            J_function = np.sum(r_nk*distance)
            if np.abs(J - J_function) <= self.e:
                break;
            J = J_function
            # print(center_denominater)
            center_points = np.sum(x_temp*np.expand_dims(r_nk, axis = 2), axis = 0)/np.expand_dims(np.sum(r_nk, axis = 0), axis = 1)
        self.max_iter = iteration
        center_temp = ones_temp * center_points
        distance = np.sum((x_temp-center_temp)**2,axis=2)
        y = np.argmin(distance,axis=1)
        centroids = center_points

        return centroids, y, self.max_iter

        


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)

            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (N,) numpy array)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels
        k_mean = KMeans(self.n_cluster, self.max_iter, self.e, self.generator)
        centroids, k_index, max_iter = k_mean.fit(x, centroid_func)
        array_dist = {i:{} for i in range(self.n_cluster)}
        for i in range(N):
            if y[i] not in array_dist[k_index[i]]:
                array_dist[k_index[i]][y[i]] = 1
            else:
                array_dist[k_index[i]][y[i]] += 1
        centroid_labels = np.zeros(self.n_cluster)
        for j in range(self.n_cluster):
            centroid_labels[j] = max(array_dist[j], key=array_dist[j].get)

        # DONOT CHANGE CODE ABOVE THIS LINE
        #raise Exception(
             #'Implement fit function in KMeansClassifier class')

        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        #assert self.centroid_labels.shape == (
            #self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        #assert self.centroids.shape == (
            #self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels
        mod_centroids = np.expand_dims(self.centroids,axis =1)
        distance = np.sum((x - mod_centroids)**2,axis = 2)
        index = np.argmin(distance,axis = 0)
        labels = self.centroid_labels[index]


        # DONOT CHANGE CODE ABOVE THIS LINE
        #raise Exception(
             #'Implement predict function in KMeansClassifier class')
        
        
        # DO NOT CHANGE CODE BELOW THIS LINE
        return np.array(labels)
        

def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors

        Return new image from the image by replacing each RGB value in image with nearest code vectors (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    # TODO
    # - comment/remove the exception
    # - implement the function
    N, M, D = image.shape
    N1, D = code_vectors.shape
    pixel = image.reshape(N * M, D)
    N2= N*M
    new_pixel = np.zeros((N2,D))

    for i in range(N2):
        d_min = np.sum((code_vectors[0] - pixel[i]) ** 2)
        d_index = 0
        for j in range(N1):
            d_curr = np.sum((code_vectors[j] - pixel[i]) ** 2)
            if d_curr < d_min:
                d_index = j
                d_min = d_curr
        new_pixel[i] = code_vectors[d_index]
    new_im = new_pixel.reshape(N, M, D)



    # DONOT CHANGE CODE ABOVE THIS LINE

    #raise Exception(
             #'Implement transform_image function')
    

    # DONOT CHANGE CODE BELOW THIS LINE
    return new_im

