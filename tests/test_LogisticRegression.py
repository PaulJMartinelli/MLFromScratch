import numpy as np
import pytest
from ML.base import BaseModel
from ML.LogisticRegression import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

# tiny separable dataset 
# less than 3 hours returns 0, 1 otherwise
X = np.array([[1], [1.5], [2], [2.5], [3], [3.5], [4], [4.5], [5]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])


def test_logistic_regression_converges_on_separable_data():
    model = LogisticRegression()
    model.fit(X, y)

    predictions = model.predict(X)

    # model should correctly classify all points on this clearly-separable data
    assert np.array_equal(predictions, y)

# ensure that calling predict() before fit() raises an error
def test_predict_before_fit_raises_error():
    model = LogisticRegression()
    X = np.array([[1], [2]])
    with pytest.raises(RuntimeError):
        model.predict(X)

    