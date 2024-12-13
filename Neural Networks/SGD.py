import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        np.random.seed(42)
        self.weights = {
            'hidden': np.random.randn(input_size, hidden_size),
            'output': np.random.randn(hidden_size, output_size)
        }

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_propagation(self, X):
        self.hidden_input = np.dot(X, self.weights['hidden'])
        self.hidden_output = self.sigmoid(self.hidden_input)

        self.output_input = np.dot(self.hidden_output, self.weights['output'])
        self.output = self.sigmoid(self.output_input)

        return self.hidden_output, self.output

    def backward_propagation(self, X, y, hidden_output, output, learning_rate):
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)

        hidden_error = output_delta.dot(self.weights['output'].T)
        hidden_delta = hidden_error * self.sigmoid_derivative(hidden_output)

        self.weights['output'] += np.outer(hidden_output, output_delta) * learning_rate
        self.weights['hidden'] += np.outer(X, hidden_delta) * learning_rate

    def learning_rate_schedule(self, gamma_0, d, t):
        return gamma_0 / (1 + (gamma_0 / d) * t)

    def train(self, X_train, y_train, hidden_size, gamma_0, d, epochs):
        input_size = X_train.shape[1]
        self.weights = {
            'hidden': np.random.randn(input_size, hidden_size),
            'output': np.random.randn(hidden_size, 1)
        }

        learning_curve = []
        for epoch in range(epochs):
            permutation = np.random.permutation(len(X_train))
            X_train = X_train[permutation]
            y_train = y_train[permutation]

            for i in range(len(X_train)):
                X = X_train[i]
                y = y_train[i]
                gamma_t = self.learning_rate_schedule(gamma_0, d, epoch * len(X_train) + i)

                hidden_output, output = self.forward_propagation(X)
                self.backward_propagation(X, y, hidden_output, output, gamma_t)

                if i % 100 == 0:
                    loss = self.compute_loss(X_train, y_train)
                    learning_curve.append(loss)
        return learning_curve

    def compute_loss(self, X, y):
        _, output = self.forward_propagation(X)
        return np.mean((y - output) ** 2)

    def predict(self, X_test):
        predictions = []
        for i in range(len(X_test)):
            _, output = self.forward_propagation(X_test[i])
            predictions.append(1 if output[0] >= 0.5 else 0)
        return np.array(predictions)

# Load data
train_data = pd.read_csv("datasets/bank-note/train.csv", header=None)
test_data = pd.read_csv("datasets/bank-note/test.csv", header=None)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

widths = [5, 10, 25, 50, 100]
gamma_0 = 0.5
d = 0.001
epochs = 100

results = []

for hidden_size in widths:
    nn = NeuralNetwork(X_train.shape[1], hidden_size, 1)
    print(f"\nTraining with hidden size: {hidden_size}")
    learning_curve = nn.train(X_train, y_train, hidden_size, gamma_0, d, epochs)

    train_error = 1 - np.mean(nn.predict(X_train) == y_train)
    test_error = 1 - np.mean(nn.predict(X_test) == y_test)

    results.append({'hidden_size': hidden_size, 'train_error': train_error, 'test_error': test_error, 'learning_curve': learning_curve})

    print(f"Training Error: {train_error:.4f} | Test Error: {test_error:.4f}")
