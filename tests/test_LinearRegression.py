import numpy as np
import pytest
from ml.base import BaseModel
from ml.LinearRegression import LinearRegression
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as SklearnLR



X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])  # basic linear relationship: y = 2 * x
def test_on_small_dataset():
    # create small dataset

    # instantiate and fit the model
    model = LinearRegression()
    model.fit(X, y)

    # Benchmark against sklearn's LinearRegression
    predictions = model.predict(X)
    print("MLFromScratch weights:", model.weights)
    print("MLFromScratch bias:", model.bias)
    sklearn_model = SklearnLR()
    sklearn_model.fit(X, y)
    print("sklearn weight:", sklearn_model.coef_)
    print("sklearn bias:", sklearn_model.intercept_)


    # Check if predictions are close(>= 0.1 tolerance) to actual values(slope of 2 and y-intercept of 0) 
    assert np.isclose(model.weights[0], 2, atol=0.1)
    assert np.isclose(model.bias, 0, atol=0.1)  

# ensure that calling predict() before fit() raises an error
def test_predict_before_fit_raises_error():
    model = LinearRegression()
    X = np.array([[1], [2]])
    with pytest.raises(RuntimeError):
        model.predict(X)

