"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd

###### Q1.1 ######
def mean_absolute_error(w, X, y):
    """
    Compute the mean absolute error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean absolute error
    """
    #####################################################
    # TODO 1: Fill in your code here #
    #####################################################
    if w is None:
        return None

    err = None
    yhat = np.dot(X , w)
    err = np.abs(np.subtract(yhat,y)).mean()
    return err

###### Q1.2 ######
def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing feature.
  - y: A numpy array of shape (num_samples, ) containing label
  Returns:
  - w: a numpy array of shape (D, )
  """
  #####################################################
  #	TODO 2: Fill in your code here #
  #####################################################		
  w = None

  xtx= np.dot(X.T ,X)
  if np.linalg.det(xtx) != 0:
      w = np.dot(np.dot(np.linalg.inv(xtx),X.T), y)
  return w

###### Q1.3 ######
def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 3: Fill in your code here #
    #####################################################
    w = None
    xtx = np.dot(X.T, X)
    while True:
        eigen= np.absolute(np.linalg.eigvals(xtx))
        if np.min(eigen) < 0.00001:
           xtx = np.add(xtx, np.identity(len(xtx))*0.1)
        else:
            w = np.dot(np.dot(np.linalg.inv(xtx), X.T), y)
            break
    return w


###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here #
  #####################################################		
    w = None
    xtx = np.dot(X.T, X)
    xtx = np.add(xtx, np.identity(len(xtx)) * 0.1)
    w = np.dot(np.dot(np.linalg.inv(xtx), X.T), y)
    return w

###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    #####################################################
    # TODO 5: Fill in your code here #
    #####################################################

    bestlambda = None
    mean_abs_err = 10000000
    power = -19
    while power < 20:
        lambda0 = 10 ** (power)
        w = regularized_linear_regression(Xtrain, ytrain, lambda0)
        err = mean_absolute_error(w, Xval, yval)
        if err < mean_abs_err:
            mean_abs_err = err
            bestlambda = lambda0
        power = power + 1
    return bestlambda



###### Q1.6 ######
def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    #####################################################
    # TODO 6: Fill in your code here #
    #####################################################		
    curr_x= X
    mapped_X = X
    i=1
    while i <power:
        curr_x = np.multiply(curr_x,X)
        mapped_X= np.concatenate((mapped_X,curr_x),axis=1)
        i=i+1
    return mapped_X


