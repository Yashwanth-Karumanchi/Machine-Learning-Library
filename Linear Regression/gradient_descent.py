import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

column_headers = ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr', 'SLUMP']
train_data = pd.read_csv("datasets/concrete/train.csv", names=column_headers)
test_data = pd.read_csv("datasets/concrete/train.csv", names=column_headers)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values.reshape(-1, 1)
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values.reshape(-1, 1)

X_train_mean = np.mean(X_train, axis=0)
X_train_std = np.std(X_train, axis=0)
X_train = (X_train - X_train_mean) / X_train_std

X_test = (X_test - X_train_mean) / X_train_std

X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

n_features = X_train.shape[1]
w = np.zeros((n_features, 1))
tolerance = 1e-6
max_iters = 10000
r = 1
min_r = 0.01

def cost_function(X, y, w):
    m = len(y)
    predictions = X.dot(w)
    return (1 / (2 * m)) * np.sum((predictions - y) ** 2)

def gradient_descent(X, y, w, r, max_iters, tolerance):
    m = len(y)
    cost_history = []
    for i in range(max_iters):
        predictions = X.dot(w)
        gradient = (1 / m) * X.T.dot(predictions - y)
        w_new = w - r * gradient
        cost = cost_function(X, y, w_new)
        cost_history.append(cost)

        if np.linalg.norm(w_new - w) < tolerance:
            print(f"Converged at iteration {i+1} with learning rate {r}")
            break

        w = w_new

    return w, cost_history

converged = False
while r >= min_r and not converged:
    print(f"Trying learning rate: {r}")
    w = np.zeros((n_features, 1))
    w, cost_history = gradient_descent(X_train, y_train, w, r, max_iters, tolerance)

    if len(cost_history) < max_iters:
        converged = True
    else:
        r /= 2

formatted_weights = ", ".join([f"{weight:.4f}" for weight in w.flatten()])
print(f"Final learned weights: {formatted_weights}")
plt.plot(cost_history)
plt.title('Cost Function vs. Iterations')
plt.xlabel('Iterations')
plt.ylabel('Cost Function')
plt.show()

test_cost = cost_function(X_test, y_test, w)
print(f"Cost function value for test data: {test_cost:.4f}")