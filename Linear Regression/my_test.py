from linear_regression import mapping_data
import json
import numpy as np
import pandas as pd


white = pd.read_csv('winequality-white', low_memory=False, sep=';').values

[N, d] = white.shape

if (mapping == True):
    maped_X = mapping_data(white[:, :-1], mapping_power)
    white = np.insert(maped_X, maped_X.shape[1], white[:, -1], axis=1)

np.random.seed(3)
    # prepare data
ridx = np.random.permutation(N)
ntr = int(np.round(N * 0.8))
nval = int(np.round(N * 0.1))
ntest = N - ntr - nval

    # spliting training, validation, and test
Xtrain = np.hstack([np.ones([ntr, 1]), white[ridx[0:ntr], 0:-1]])

ytrain = white[ridx[0:ntr], -1]

Xval = np.hstack([np.ones([nval, 1]), white[ridx[ntr:ntr + nval], 0:-1]])
yval = white[ridx[ntr:ntr + nval], -1]

Xtest = np.hstack([np.ones([ntest, 1]), white[ridx[ntr + nval:], 0:-1]])
ytest = white[ridx[ntr + nval:], -1]
if (non_invertible == True):
        N, D = Xtrain.shape
        np.random.seed(4)
        random_row = np.random.randint(N)
        random_col = np.random.randint(D)

        Xtrain[:, random_col] = 0
        Xtrain[random_row, :] = 0

print( Xtrain, ytrain, Xval, yval, Xtest, ytest)



bestlambda = None
    mean_abs_err = 10000000
    power = -19
    while power<20:
        lambda0 = 10 ** (power)
        w=regularized_linear_regression(Xtrain,ytrain,lambda0)
        err= mean_absolute_error(w, Xval, yval)
        if err<mean_abs_err:
            mean_abs_err = err
            bestlambda = lambda0
        power=power+1
    return bestlambda


