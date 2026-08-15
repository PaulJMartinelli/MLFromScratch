import numpy as np
import pandas as pd
from ML.base import BaseModel
from collections import Counter


class Node:
    def __init__(self, feature_index = None, threshold = None, info_gain = None, left = None, right = None, value = None):
        # decision node:
        self.feature_index = feature_index
        self.threshold = threshold
        self.info_gain = info_gain
        self.left = left
        self.right = right

        # leaf node:
        self.value = value

class ClassificationTree(BaseModel):
    def __init__(self, min_samples_split=2, max_depth=2):

        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    # dataset becomes a subset of the original as we recursively traverse the tree
    def build_tree(self, dataset, curr_depth=0):
        X, y = dataset[:, :-1], dataset[:, -1]
        n_samples, n_features = X.shape

        if n_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            # decision node
            best_split = self.best_split(dataset, n_features)
            if best_split["info_gain"] > 0:
                left_node = self.build_tree(best_split["left_dataset"], curr_depth + 1)
                right_node = self.build_tree(best_split["right_dataset"], curr_depth + 1)
                return Node(best_split["feature_index"], best_split["threshold"], best_split["info_gain"], left_node, right_node)

        # leaf node
        leaf_value = Counter(y).most_common(1)[0][0]
        return Node(value=leaf_value)

    # best split so we can determine most telling feature at each node
    def best_split(self, dataset, n_features):
        
        best_split = {'feature_index': None, 'threshold': None, 'info_gain': -1.0, 'left_dataset': None, 'right_dataset': None}

        for feature_index in range(n_features):
            feature_values = dataset[:, feature_index]
            thresholds = np.unique(feature_values)

            for threshold in thresholds:
                # split into left and right subsets around the threshold
                left_dataset, right_dataset = self.split(dataset, feature_index, threshold)
                if len(left_dataset) > 0 and len(right_dataset) > 0:
                    parent_y, left_y, right_y = dataset[:, -1], left_dataset[:, -1], right_dataset[:, -1]

                    info_gain = self.information_gain(parent_y, left_y, right_y)
                    if info_gain > best_split["info_gain"]:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["info_gain"] = info_gain
                        best_split["left_dataset"] = left_dataset
                        best_split["right_dataset"] = right_dataset

        return best_split

    # divide dataset into left and right subsets based on threshold for a given feature
    def split(self, dataset, feature_index, threshold):
        left_dataset = np.array([row for row in dataset if row[feature_index] <= threshold])
        right_dataset = np.array([row for row in dataset if row[feature_index] > threshold])

        return left_dataset, right_dataset

    # find how much a given split reduces impurity
    def information_gain(self, parent_y, left_y, right_y):
        left_weight = len(left_y) / len(parent_y)
        right_weight = len(right_y) / len(parent_y)

        information_gain = self.gini(parent_y) - (left_weight * self.gini(left_y) + right_weight * self.gini(right_y))
        return information_gain

    def gini(self, y):
        class_labels = np.unique(y)
        gini = 1.0 

        for label in class_labels:
            p_label = len(y[y == label]) / len(y)
            gini -= p_label ** 2

        return gini

    def fit(self, X, y):
        full_dataset = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        self.root = self.build_tree(full_dataset)

    def predict(self, X):
        predictions = [self.predict_classification(x, self.root) for x in X]
        return np.array(predictions)

    def predict_classification(self, row, node):
        # base case: if we reach a leaf node
        if node.value is not None:
            return node.value

        # if we are at a decision node, traverse based on feature value
        feature_value = row[node.feature_index]
        if feature_value <= node.threshold:
            return self.predict_classification(row, node.left)
        else:
            return self.predict_classification(row, node.right)