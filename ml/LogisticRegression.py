import numpy as np
from ML.base import BaseModel


class LogisticRegression(BaseModel):
    
    def __init__(self):
        super().__init__()
        self.learning_rate = 0.01
        self.epochs = 5000
        self.weights = None
        self.bias = None
        # store each loss value in array to plot loss vs epoch later 
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # initialize slopes and y-intercept as 0
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            #linear combination
            linear_model = np.dot(X, self.weights) + self.bias
            # turn into probabilities (between 0 and 1)
            y_prediction = self.sigmoid(linear_model)
            loss = self.compute_loss(y, y_prediction)
            self.loss_history.append(loss)
            dw, db = self.compute_gradients(X, y, y_prediction)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        # mark fitted before predicting
        self._is_fitted = True

    #sigmoid for np array
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, X):
        self._check_is_fitted()
        linear_model = np.dot(X, self.weights) + self.bias
        y_prediction = self.sigmoid(linear_model)
        # convert probabilities to binary predictions (0 or 1)
        return np.round(y_prediction).astype(int)
            
    def compute_gradients(self, X, y_actual, y_prediction):
        n_samples = X.shape[0]
        residual = y_prediction - y_actual
        dw = (1/n_samples) * np.dot(X.T, residual)
        db = (1/n_samples) * np.sum(residual)
        return dw, db

    
    def compute_loss(self, y_actual, y_prediction):
        # binary cross-entropy loss
        epsilon = 1e-15  # small clip to avoid log(0)
        y_prediction = np.clip(y_prediction, epsilon, 1 - epsilon)
        n_samples = y_actual.shape[0]
        loss = -(1/n_samples) * np.sum(y_actual * np.log(y_prediction) + (1 - y_actual) * np.log(1 - y_prediction))
        return loss