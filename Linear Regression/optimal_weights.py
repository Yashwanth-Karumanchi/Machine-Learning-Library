import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

column_headers = ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr', 'SLUMP']
train_data = pd.read_csv("datasets/concrete/train.csv", names=column_headers)
test_data = pd.read_csv("datasets/concrete/test.csv", names=column_headers)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values.reshape(-1, 1)
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values.reshape(-1, 1)

X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# Using the Normal Equation: w_optimal = (X^T X)^(-1) X^T y
XT_X = X_train.T.dot(X_train)  # X^T X
XT_y = X_train.T.dot(y_train)  # X^T y
w_optimal = np.linalg.inv(XT_X).dot(XT_y)  # (X^T X)^(-1) X^T y

formatted_weights = ", ".join([f"{weight:.4f}" for weight in w_optimal.flatten()])
print(f"Optimal weight vector (Normal Equation): {formatted_weights}")

m_test = len(y_test)
predictions_test = X_test.dot(w_optimal)
cost_test = (1 / (2 * m_test)) * np.sum((predictions_test - y_test) ** 2)

print(f"Cost function value for test data (Optimal weights): {cost_test:.4f}")