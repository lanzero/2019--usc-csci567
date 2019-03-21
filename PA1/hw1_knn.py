from __future__ import division, print_function

from typing import List

import numpy as np
import scipy

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:

    def __init__(self, k: int, distance_function):
        self.k = k
        self.distance_function = distance_function

    #TODO: save features and lable to self
    def train(self, features: List[List[float]], labels: List[int]):
        # features: List[List[float]] a list of points
        # labels: List[int] labels of features
        self.features = features
        self.labels = np.array(labels,dtype = int).tolist()
        return

    #TODO: predict labels of a list of points
    def predict(self, features: List[List[float]]) -> List[int]:
        # features: List[List[float]] a list of points
        # return: List[int] a list of predicted labels
        answer = []
        for i in features:
            results = np.array(self.get_k_neighbors(i),dtype=int)
            a = np.argmax(np.bincount(results))
            answer.append(a)
        return answer
            
    #TODO: find KNN of one point
    def get_k_neighbors(self, point: List[float]) -> List[int]:
        # point: List[float] one example
        # return: List[int] labels of K nearest neighbor
        distance = []
        result = []
        for i in self.features:
            distance.append(self.distance_function(point,i))
        distance_labels = np.array(list(zip(distance,self.labels)),dtype = [('distance',float),('label',int)])
        sortedDistance = np.sort(distance_labels, order = 'distance')
        for i in range(self.k):
            result.append(sortedDistance[i][1].tolist())
        return result


if __name__ == '__main__':
    print(np.__version__)
    print(scipy.__version__)
