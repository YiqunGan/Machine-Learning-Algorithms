import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################
import math

# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """


    product_sum=[]
    for i,j in zip(real_labels, predicted_labels):
        product_sum.append(i*j)
    F1_score=(2*sum(product_sum))/(sum(real_labels)+sum(predicted_labels))

    return F1_score
    raise NotImplementedError








class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        distance =0
        difference = (np.abs(np.subtract(point1,point2)))**3
        distance= (sum(difference))**(1/3)

        return distance
        raise NotImplementedError

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        difference = np.subtract(point1,point2)
        return np.sqrt(np.inner(difference,difference))
        raise NotImplementedError

    @staticmethod
    # TODO
    def inner_product_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        return np.inner(point1,point2)
        raise NotImplementedError

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        distance=1-np.array(np.inner(point1,point2))/(np.linalg.norm(point1)*np.linalg.norm(point2))
        return distance
        raise NotImplementedError

    @staticmethod
    # TODO
    def gaussian_kernel_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """

        return -np.exp(-(np.inner(np.subtract(point1,point2),np.subtract(point1,point2)))/2)
        raise NotImplementedError


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you should try different distance function you implemented in part 1.1, and find the best k.
        Use k range from 1 to 30 and increment by 2. Use f1-score to compare different models.

        :param distance_funcs: dictionary of distance functions you must use to calculate the distance.
            Make sure you loop over all distance functions for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val:  List[List[int]] Validation data set will be used on your KNN predict function to produce
            predicted labels and tune k and distance function.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_function and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function and self.best_model respectively.
        NOTE: self.best_scaler will be None

        NOTE: When there is a tie, choose model based on the following priorities:
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance fuction, choose model which has a less k.
        """
        
        # You need to assign the final values to these variables
        best_f1_score = -1
        self.best_k = None
        self.best_distance_function = None
        self.best_model = None
        dist_priority = {'euclidean': 1, 'minkowski': 2, 'gaussian': 3, 'inner_prod': 4, 'cosine_dist': 5}
        for key, dist in distance_funcs.items():
            k = 1
            while k <= 30:
                curr_knn = KNN(k,dist)
                curr_knn.train(x_train,y_train)
                y_val_predict = curr_knn.predict(x_val)
                val_f1_score = f1_score(y_val,y_val_predict)
                if best_f1_score < val_f1_score:
                    self.best_k = k
                    best_f1_score =val_f1_score
                    self.best_distance_function = key
                    self.best_model = curr_knn
                if best_f1_score == val_f1_score:
                    if dist_priority[key] < dist_priority[self.best_distance_function]:
                        self.best_k = k
                        best_f1_score = val_f1_score
                        self.best_distance_function = key
                        self.best_model = curr_knn
                    if dist_priority[key] == dist_priority[self.best_distance_function]:
                        if k < self.best_k:
                            self.best_k = k
                            best_f1_score = val_f1_score
                            self.best_distance_function = key
                            self.best_model = curr_knn
                k = k+2
        return self.best_k, self.best_distance_function, self.best_model
        raise NotImplementedError

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is similar to Part 1.3 except that before passing your training and validation data to KNN model to
        tune k and disrance function, you need to create the normalized data using these two scalers to transform your
        data, both training and validation. Again, we will use f1-score to compare different models.
        Here we have 3 hyperparameters i.e. k, distance_function and scaler.

        :param distance_funcs: dictionary of distance funtions you use to calculate the distance. Make sure you
            loop over all distance function for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param scaling_classes: dictionary of scalers you will use to normalized your data.
        Refer to test.py file to check the format.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data set you will use on your KNN predict function to produce predicted
            labels and tune your k, distance function and scaler.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function, self.best_scaler and self.best_model respectively

        NOTE: When there is a tie, choose model based on the following priorities:
        For normalization, [min_max_scale > normalize];
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance function, choose model which has a less k.
        """
        # You need to assign the final values to these variables
        best_f1_score = -1
        self.best_k = None
        self.best_distance_function = None
        self.best_model = None
        dist_priority = {'euclidean': 1, 'minkowski': 2, 'gaussian': 3, 'inner_prod': 4, 'cosine_dist': 5}
        normal_priority = {'min_max_scale': 1, 'normalize': 2}
        self.best_scaler = None
        for name, scaler in scaling_classes.items():
            scaler = scaler()
            x_train = scaler(x_train)
            x_val = scaler(x_val)
            for key, dist in distance_funcs.items():
                k = 2
                while k <= 30:
                    curr_knn = KNN(k, dist)
                    curr_knn.train(x_train, y_train)
                    y_val_predict = curr_knn.predict(x_val)
                    val_f1_score = f1_score(y_val, y_val_predict)
                    if best_f1_score < val_f1_score:
                        self.best_k = k
                        best_f1_score = val_f1_score
                        self.best_distance_function = key
                        self.best_scaler = name
                        self.best_model = curr_knn
                    elif best_f1_score == val_f1_score:
                        if normal_priority[name] < normal_priority[self.best_scaler]:
                            self.best_k = k
                            best_f1_score = val_f1_score
                            self.best_distance_function = key
                            self.best_scaler = name
                            self.best_model = curr_knn
                        elif normal_priority[name] == normal_priority[self.best_scaler]:
                            if dist_priority[key] < dist_priority[self.best_distance_function]:
                                self.best_k = k
                                best_f1_score = val_f1_score
                                self.best_distance_function = key
                                self.best_scaler = name
                                self.best_model = curr_knn
                            elif dist_priority[key] == dist_priority[self.best_distance_function]:
                                if k < self.best_k:
                                    self.best_k = k
                                    best_f1_score = val_f1_score
                                    self.best_distance_function = key
                                    self.best_scaler = name
                                    self.best_model = curr_knn
                    k = k + 2
            return self.best_k, self.best_distance_function, self.best_model
        raise NotImplementedError





class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """

        normalization = []
        sum = np.linalg.norm(features, axis = 1)
        for i, j in zip(features, sum):
            if j == 0:
                normalization.append(np.zeros(len(i)))
            else:
                normalization.append(np.true_divide(i, j))

        return normalization





class MinMaxScaler:
    """
    Please follow this link to know more about min max scaling
    https://en.wikipedia.org/wiki/Feature_scaling
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
    will be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler1 = MinMaxScale()
        train_features_scaled = scaler1(train_features)
        # train_features_scaled should be equal to [[0, 1], [1, 0]]

        test_features_scaled = scaler1(test_features)
        # test_features_scaled should be equal to [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        pass
        self.max_value = []
        self.min_value = []



    def __call__(self, features):
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]

        :param features: List[List[float]]
        :return: List[List[float]]
                for i, j in zip(transpose_features, difference):
            if j == 0:
                after_scaler.append(np.zeros(len(i)))
            else:
                after_scaler.append(np.true_divide(i-self.min_value[j], j))
        after_scaler = np.transpose(after_scaler)
        """

        after_scaler = []
        if len(self.max_value) == 0:
            self.max_value = np.max(features, axis=0)
            self.min_value = np.min(features, axis=0)
        difference = self.max_value - self.min_value
        transpose_features = np.transpose(features)
        for i in range(len(transpose_features)):
            if difference[i] == 0:
                after_scaler.append(np.zeros(len(transpose_features[0])))
            else:
                after_scaler.append(np.true_divide(transpose_features[i] - self.min_value[i], difference[i]))
        after_scaler = np.transpose(after_scaler)
        return after_scaler

