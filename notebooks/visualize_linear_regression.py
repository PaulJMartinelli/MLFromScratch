import numpy as np
import matplotlib.pyplot as plt
from ML.LinearRegression import LinearRegression

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])  # y = 2x

model = LinearRegression()
model.fit(X, y)

print("Weights:", model.weights)
print("Bias:", model.bias)

plt.plot(model.loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training Loss Curve for Linear Regression')
plt.show()