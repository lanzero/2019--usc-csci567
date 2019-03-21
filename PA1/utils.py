import numpy as np
from typing import List
from hw1_knn import KNN
import copy
from sklearn.metrics import accuracy_score
# TODO: Information Gain function
def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    ave_en = S #the whole information gain
    tot = 0.0
    total_sum = 0
    for i in branches:
        for j in i:
            total_sum += j
    if total_sum == 0:
        return S
    for i in branches:
        h = 0.0 # entropy for each value, there will be len(branches) entropies
        esum = np.sum(i)
        if esum == 0:
            continue
        for k in i:
            if k == 0:
                h1 = 0
            else:
                h1 = -k/esum*np.log2(k/esum)
            h += h1
        tot += h*esum/total_sum
    return round(ave_en-tot,10)


# TODO: implement reduced error prunning function, pruning your tree on this function
def reduced_error_prunning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List
    y_est_test = decisionTree.predict(decisionTree.root_node.features)
    decisionTree.prun(X_test, y_test)
    return 


# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')


#TODO: implement F1 score
def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    assert len(real_labels) == len(predicted_labels)
    real = np.asarray(real_labels)
    pred = np.asarray(predicted_labels)
    l = len(real_labels)
    return 2*np.dot(real,pred)/(np.sum(real)+np.sum(pred))
#TODO:
def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    return np.linalg.norm(np.asarray(point1)-np.asarray(point2))


#TODO:
def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    return np.dot(point1,point2)


#TODO:
def gaussian_kernel_distance(point1: List[float], point2: List[float]) -> float:
    return -np.exp(-0.5*np.dot(np.asarray(point1)-np.asarray(point2), np.asarray(point1)-np.asarray(point2)))


#TODO:
def cosine_sim_distance(point1: List[float], point2: List[float]) -> float:
    return 1-np.dot(point1,point2)/(np.linalg.norm(point1)*np.linalg.norm(point2))


# TODO: select an instance of KNN with the best f1 score on validation dataset
def model_selection_without_normalization(distance_funcs, Xtrain, ytrain, Xval, yval):
    # distance_funcs: dictionary of distance funtion
    # Xtrain: List[List[int]] train set
    # ytrain: List[int] train labels
    # Xval: List[List[int]] validation set
    # yval: List[int] validation labels
    # return best_model: an instance of KNN
    # return best_k: best k choosed for best_model
    # return best_func: best function choosed for best_model
    allknn = []
    performance = []
    function = []
    for key,value in distance_funcs.items():
        for k in range(1,min(30,len(Xtrain)),2):
            knn = KNN(k,value)
            knn.train(Xtrain,ytrain)
            answer = knn.predict(Xval)
            score = f1_score(yval,answer)
            allknn.append(knn)
            function.append(key)
            performance.append(score)
            #print(k,',',key,',',answer,',',yval,',',score)
    result = allknn[np.argmax(np.array(performance))]
    best_function = function[np.argmax(np.array(performance))]
    return result, result.k, best_function


# TODO: select an instance of KNN with the best f1 score on validation dataset, with normalized data
def model_selection_with_transformation(distance_funcs, scaling_classes, Xtrain, ytrain, Xval, yval):
    # distance_funcs: dictionary of distance funtion
    # scaling_classes: diction of scalers
    # Xtrain: List[List[int]] train set
    # ytrain: List[int] train labels
    # Xval: List[List[int]] validation set
    # yval: List[int] validation labels
    # return best_model: an instance of KNN
    # return best_k: best k choosed for best_model
    # return best_func: best function choosed for best_model
    # return best_scaler: best function choosed for best_model
    allknn = []
    performance = []
    scaler = []
    function = []
    for kkey, vvalue in scaling_classes.items():
        for key,value in distance_funcs.items():
            for k in range(1,min(30,len(Xtrain)),2):
                knn = KNN(k,value)
                cur_sca = vvalue()
                knn.train(cur_sca(Xtrain),ytrain)
                answer = knn.predict(cur_sca(Xval))
                score = f1_score(yval,answer)
                allknn.append(knn)
                function.append(key)
                scaler.append(kkey)
                performance.append(score)
                #print(k,',',key,',',answer,',',yval,',',score)
    result = allknn[np.argmax(np.array(performance))]
    best_sca = scaler[np.argmax(np.array(performance))]
    best_function = function[np.argmax(np.array(performance))]
    return result, result.k, best_function, best_sca


class NormalizationScaler:
    def __init__(self):
        pass

    #TODO: normalize data
    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        result = []
        for i in features:
            scaler = np.sqrt(np.dot(i,i))
            if scaler == 0:
                result.append([0] * len(i))
            else:
                result.append((np.array(i)/scaler).tolist())
        return result
            


class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]
    """
    def __init__(self):
        self.first = 0

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        #TODO: check for the first call 
        answer = []
        if self.first == 0:
            self.first = 1
            self.max = np.array([],dtype = float)
            self.min = np.array([],dtype = float)
            self.max = np.max(features,axis = 0)
            self.min = np.min(features,axis = 0)
        for i in features:
            result = np.array([],dtype = float)
            for k, mi, ma in zip(i, self.min, self.max):
                if mi != ma:
                    result = np.append(result, (k - mi)/(ma - mi))
                else:
                    result = np.append(result, k)
            
            answer.append(result.tolist())
        return answer



