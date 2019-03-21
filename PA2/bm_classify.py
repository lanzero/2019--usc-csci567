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

    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0
        ############################################
        new_y = np.where(y == 1, y, y-1)
        for i in range(max_iterations):
            inte_w = np.zeros(D)
            inte_b = 0.0
            #for j in range(N):
            #    if new_y[j]*(np.matmul(w,X[j])+b) <= 0:
            #        inte_w += new_y[j]*X[j]
            #        inte_b += new_y[j]
            z = np.multiply(np.matmul(X,w),new_y)+np.multiply(b*np.ones(N),new_y)
            inte = np.multiply(np.where(z<=0,z-z+1,z-z),new_y)
            inte_w = np.matmul(inte.T,X)
            inte_b = np.dot(inte,np.ones(N))
            w += 0.5* inte_w/N
            b += 0.5* inte_b/N

    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0
        ############################################
        new_y = np.where(y == 1, y, y-1)
        z = np.array([])
        for i in range(max_iterations):
            temp = np.zeros(D)
            t1 = 0.0
            z = np.multiply(new_y, np.matmul(X,w)+b*np.ones(N)) 
            sig = sigmoid(-z)
           # for j in range(N):
           #     temp += sig[j]*new_y[j]*X[j]
           #     t1 += sig[j]*new_y[j]
            temp = np.multiply(sig,new_y)
            w += 0.5*np.matmul(temp,X)/N
            b += 0.5*np.dot(temp,np.ones(N))/N
            
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
    value = z
    ############################################
    if np.isscalar(value):
        value = 1/(1 + np.exp(-z))
    else:
        inte = np.ones(len(z))
        value = inte/(inte + np.exp(-z))
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
        z = np.matmul(X,w) + b*np.ones(N)
        for i in np.where(z>0):
            preds[i] = 1
        

    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)
        ############################################
        z = np.matmul(X,w) + b*np.ones(N)
        the = sigmoid(z)
        for i in np.where(the>0.5):
            preds[i] = 1

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
        ############################################
        a = np.array(range(N))
        for i in range(max_iterations):
            j = np.random.choice(a)
            yn = y[j]
            xn = X[j]
            p = softmax(np.matmul(w,xn)+b,0)
            p[y[j]] -= 1
            w -= 0.5*np.matmul(np.mat(p).T,np.mat(xn))
            b -= 0.5*p
    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        ############################################
        for i in range(max_iterations):
            f = softmax(np.matmul(w,X.T) + np.matmul(np.mat(b).T,np.mat(np.ones(N))),1)
            print(f.shape)
            onehot_y = np.zeros((N,C))
            onehot_y[np.arange(N),y] = 1
            w -= 0.5*np.matmul((f - onehot_y.T),X)/N
            b -= 0.5*np.dot(np.asarray((f - onehot_y.T)),np.ones(N))/N
            
    else:
        raise "Type of Gradient Descent is undefined."
    

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b

def softmax(x,s):
    if s == 0:                    
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    else:
        e_max = np.max(x,axis = 0)
        e_x = np.exp(x - np.matmul(np.mat(np.ones(len(x))).T,e_max))            
        return e_x / np.matmul(np.mat(np.ones(len(x))).T,np.mat(np.sum(e_x,axis = 0)))   


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
    ############################################
    p = np.matmul(w,X.T)
    preds = np.argmax(p,axis = 0)*1.0
    assert preds.shape == (N,)
    return preds




        