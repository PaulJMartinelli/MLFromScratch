import numpy as np
from collections import Counter
from ML.base import BaseModel

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

class KNN(BaseModel):
    def __init__(self, k = 3):
        self.k = k

    def fit(self, X, y):   
        self.X_train = X
        self.y_train = y

    # take new points and predict their classes
    def predict(self, new_points):
        predictions = [self.predict_class(new_point) for new_point in new_points]
        return np.array(predictions)
        for point in new_points:
            predictions.append(self.predict_class(point))
        return np.array(predictions)

    def predict_class(self, new_point):
        # find distance of every point in training set to new point
        distances = [euclidean_distance(new_point, x) for x in self.X_train]
        # find first k nearest neighbors and return the most common label 
        k_nearest_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common