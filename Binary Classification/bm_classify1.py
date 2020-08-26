import numpy as np


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2


    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0

    y_modified  = np.where(y == 0, -1, y)

    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #

        w = np.zeros(D)
        b = 0

        for i in range(max_iterations):
            z = y_modified * (np.dot(X, w) + b)
            z = np.int32(z <= 0)
            update_b = y_modified*z
            update_w = (update_b.T).dot(X)
            w = w + (step_size / N) * update_w
            b = b + (step_size / N) * np.sum(update_b)



        ############################################
        

    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0
        ############################################
        for i in range(max_iterations):
            z = y_modified * (np.dot(X, w) + b)
            prob = (1/(np.exp(z)+1))*y_modified
            update_b = np.sum(prob)
            update_w = (prob.T).dot(X)
            w = w + (step_size / N) * update_w
            b = b + (step_size / N) * update_b


    else:
        raise "Loss Function is undefined."

    assert w.shape == (D,)
    return w, b

def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    value =1/(1+np.exp(z))


    ############################################
    
    return value

def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape
    
    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)
        ############################################
        z = np.dot(X,w)+b
        preds = np.int32(z>0)


    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)
        ############################################
        z = np.dot(X,w)+b
        prob = sigmoid(-z)
        preds = np.int32(prob>0.5)

    else:
        raise "Loss Function is undefined."
    

    assert preds.shape == (N,) 
    return preds



def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42)
    if gd_type == "sgd":
        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        n = np.random.choice(N)
        x_n = X[n]
        y_n = y[n]
        z = np.dot(x_n,w.T)+b
        denominater = np.sum(np.exp(z))
        numerator = np.exp(z)
        update_b = numerator/denominater
        update_b[y_n] = update_b[y_n]-1
        update_w =np.dot(update_b.reshape(C,1),x_n.reshape(1,D))
        w= w - step_size*update_w
        b=b- step_size*update_b
        ############################################
        

    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        for i in range(max_iterations):
            x_w = np.dot(X,w.T)+b
            exp_x_w = np.exp(x_w)
            denominater = np.sum(exp_x_w,axis=1,keepdims=True)
            onehot = np.zeros([N, C])
            for i, j in enumerate(y):
                onehot[i][j] = 1
            gradient = exp_x_w/denominater - onehot
            update_w = np.dot(gradient.T,X)
            update_b = np.sum(gradient,axis=0)
            w= w- step_size/N * update_w
            b=b- step_size/N * update_b

        ############################################
        

    else:
        raise "Type of Gradient Descent is undefined."
    

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D 
    - b: bias terms of the trained multinomial classifier, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
    preds = np.zeros(N)
    z = np.dot(X,w.T)+b
    for i in range(N):
        preds[i-1]= np.argmax(z[i-1,:])

    ############################################

    assert preds.shape == (N,)
    return preds




        