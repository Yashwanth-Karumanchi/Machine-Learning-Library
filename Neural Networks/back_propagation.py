import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class NeuralNetwork:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        np.random.seed(42)
        self.weights = {
            'hidden1': np.random.randn(input_size, hidden1_size) * 0.01,
            'hidden2': np.random.randn(hidden1_size, hidden2_size) * 0.01,
            'output': np.random.randn(hidden2_size, output_size) * 0.01
        }

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_propagation(self, X):
        self.hidden1_input = np.dot(X, self.weights['hidden1'])
        self.hidden1_output = self.sigmoid(self.hidden1_input)

        self.hidden2_input = np.dot(self.hidden1_output, self.weights['hidden2'])
        self.hidden2_output = self.sigmoid(self.hidden2_input)

        self.output_input = np.dot(self.hidden2_output, self.weights['output'])
        self.output = self.sigmoid(self.output_input)

        return self.hidden1_output, self.hidden2_output, self.output

    def backward_propagation(self, X, y, learning_rate):
        output_error = y - self.output
        output_delta = output_error * self.sigmoid_derivative(self.output)

        hidden2_error = np.dot(output_delta, self.weights['output'].T)
        hidden2_delta = hidden2_error * self.sigmoid_derivative(self.hidden2_output)

        hidden1_error = np.dot(hidden2_delta, self.weights['hidden2'].T)
        hidden1_delta = hidden1_error * self.sigmoid_derivative(self.hidden1_output)

        self.weights['output'] += np.outer(self.hidden2_output, output_delta) * learning_rate
        self.weights['hidden2'] += np.outer(self.hidden1_output, hidden2_delta) * learning_rate
        self.weights['hidden1'] += np.outer(X, hidden1_delta) * learning_rate

    def train(self, X_train, y_train, learning_rate, epochs):
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(len(X_train)):
                X = X_train[i]
                y = y_train[i]
                self.forward_propagation(X)
                self.backward_propagation(X, y, learning_rate)
                epoch_loss += np.mean((y - self.output) ** 2)

    def predict(self, X_test):
        predictions = []
        for i in range(len(X_test)):
            X = X_test[i]
            _, _, output = self.forward_propagation(X)
            predictions.append(1 if output[0] >= 0.5 else 0)
        return np.array(predictions)

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

# Initialize and train neural network
input_size = X_train.shape[1]
hidden1_size = 8
hidden2_size = 8
output_size = 1
learning_rate = 0.01
epochs = 200

nn = NeuralNetwork(input_size, hidden1_size, hidden2_size, output_size)
nn.train(X_train, y_train, learning_rate, epochs)

# Test neural network
predictions = nn.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Accuracy on test set: {accuracy * 100:.2f}%")