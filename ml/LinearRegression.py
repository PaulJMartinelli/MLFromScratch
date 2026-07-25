import numpy as np
from ml.base import BaseModel
from ml.utils.metrics import mse

class LinearRegression(BaseModel):
   
   def __init__(self):
        super().__init__()
        self.learning_rate = 0.01
        self.epochs = 1000
        self.weights = None
        self.bias = None
        # store each loss value in array to plot loss vs epoch later 
        self.loss_history = []

   # X = input features(in matrix form), y = target values
   def fit(self, X, y):
       n_samples, n_features = X.shape
       # Initialize slopes and y-intercept as 0
       self.weights = np.zeros(n_features)
       self.bias = 0

      # gradient descent (1000 epochs, learning rate 0.01)
       for epoch in range(self.epochs):
         y_prediction = np.dot(X, self.weights) + self.bias
         # compute gradients
         dw, db = self.compute_gradients(X, y, y_prediction)
         # update weights and bias
         self.weights -= self.learning_rate * dw
         self.bias -= self.learning_rate * db
         # compute and store loss(MSE) for this epoch
         loss = mse(y, y_prediction)
         self.loss_history.append(loss)

      # mark fitted before predicting
       self._is_fitted = True 

   def predict(self, X):
         self._check_is_fitted()
         return np.dot(X, self.weights) + self.bias


   # compute the gradients of the loss function
   def compute_gradients(self, X, y_actual, y_prediction):
       n_samples = X.shape[0]
       residual = y_actual - y_prediction
       dw = (-2/n_samples) * np.dot(X.T, residual)
       db = (-2/n_samples) * np.sum(residual)
       return dw, db
   
