import numpy as np
import utils as Util

class DecisionTree():
    
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        self.feature_dim = len(features[0])
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()
        return
    
    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
            # print ("feature: ", feature)
            # print ("pred: ", pred)
        return y_pred
    
    def prun(self, X_val, y_val):
        self.root_node.prun(X_val, y_val)
        return


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
                # splitable is false when all features belongs to one class or no more examples
        if len(np.unique(labels)) < 2 or len(self.features[0]) < 1:
            self.splittable = False
        else:
            self.splittable = True
            
        self.isleaf = False
        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    #TODO: try to split current node
    def split(self):
        branches = []
        IG = []
        #get entropy for this node
        labels_count = np.bincount(self.labels)
        h = 0.0
        esum = np.sum(labels_count)
        if esum == 0:
            h = 0.0
        else:
            for k in labels_count:
                if k == 0:
                    h1 = 0
                else:
                    h1 = -k/esum*np.log2(k/esum)
                h += h1
        #get IG
        for i in range(len(self.features[0])):
            result1 = []
            for j in np.unique(np.transpose(self.features)[i]):
                labels = []
                for ind,k in enumerate(np.transpose(self.features)[i]):
                    if(j == k):
                        labels.append(self.labels[ind])
                result1.append(np.bincount(labels).tolist())
            branches.append(result1)
        for i in branches:
            IG.append(Util.Information_Gain(h,i))
        self.dim_split = np.argmax(IG)
        self.feature_uniq_split = np.unique(np.transpose(self.features)[self.dim_split]).tolist()
        for i in np.unique(np.transpose(self.features)[self.dim_split]):
            feature = []
            label = []
            for index, j in enumerate(self.features):
                if i == j[self.dim_split]:
                    inter = list(j)
                    inter.pop(self.dim_split)
                    feature.append(inter)
                    label.append(self.labels[index])
            node = TreeNode(feature,label,len(np.unique(label)))
            if node.splittable:
                node.split()
            self.children.append(node)
        return 
    

    # TODO: predict the branch or the class
    def predict(self, feature):
        # feature: List[any]
        # return: int
        if not self.splittable:
            self.isleaf = True
            return self.cls_max    
        else:
            if any(i == feature[self.dim_split] for i in self.feature_uniq_split):
                ind = self.feature_uniq_split.index(feature[self.dim_split])
                temper = list(feature)
                temper.pop(self.dim_split)
                return self.children[ind].predict(temper)
            else:
                self.isleaf = True
                return self.cls_max
    def prun(self, X_val, y_val):
        acur1 = np.bincount(y_val)[self.cls_max]
        if not self.isleaf:
            acur2 = 0.0
            for ind,i in enumerate(self.children):
                Xv = []
                yv = []
                for index, j in enumerate(X_val):
                    if self.feature_uniq_split[ind] == j[self.dim_split]:
                        inter = list(j)
                        inter.pop(self.dim_split)
                        Xv.append(inter)
                        yv.append(y_val[index])
                if yv == []:
                    continue
                acur2 += i.prun(Xv,yv)
            if(acur1 > acur2):
                self.children = []
                self.splittable = False
                self.isleaf = True
                return acur1
            else:
                return acur2
        else:
            return acur1
            
        
        
        
        
        
