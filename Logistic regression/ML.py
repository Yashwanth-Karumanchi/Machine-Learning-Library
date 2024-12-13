import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

class LogisticRegressionSGD:
    def __init__(self, input_size):
        self.weights = np.zeros(input_size)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_gradient(self, X, y):
        predictions = self.sigmoid(np.dot(X, self.weights))
        gradient = -np.dot(X.T, (y - predictions))
        return gradient

    def compute_loss(self, X, y):
        predictions = self.sigmoid(np.dot(X, self.weights))
        likelihood = y * np.log(predictions + 1e-9) + (1 - y) * np.log(1 - predictions + 1e-9)
        return -np.mean(likelihood)

    def train(self, X, y, gamma_0, d, epochs):
        losses = []
        for epoch in range(epochs):
            permutation = np.random.permutation(len(X))
            X = X[permutation]
            y = y[permutation]
            for t in range(len(X)):
                learning_rate = gamma_0 / (1 + (gamma_0 / d) * (epoch * len(X) + t))
                gradient = self.compute_gradient(X[t:t+1], y[t:t+1])
                self.weights -= learning_rate * gradient
            loss = self.compute_loss(X, y)
            losses.append(loss)
        return losses

    def predict(self, X):
        return (self.sigmoid(np.dot(X, self.weights)) >= 0.5).astype(int)

# Load and preprocess data
train_data = pd.read_csv("datasets/bank-note/train.csv", header=None)
test_data = pd.read_csv("datasets/bank-note/test.csv", header=None)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

variances = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
gamma_0 = 0.5
d = 1e-2
epochs = 100

results = []

for variance in variances:
    model = LogisticRegressionSGD(X_train.shape[1])
    print(f"\nTraining with variance: {variance}")
    losses = model.train(X_train, y_train, gamma_0, d, epochs)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_error = np.mean(train_predictions != y_train)
    test_error = np.mean(test_predictions != y_test)

    results.append({
        'Variance': variance,
        'Train Error': train_error,
        'Test Error': test_error
    })

    print(f"Train Error: {train_error:.4f} | Test Error: {test_error:.4f}")

# Report results
table = tabulate(results, headers="keys", tablefmt="grid")
print(table)
