import pandas as pd
import numpy as np

class AveragePerceptron:
    def __init__(self, learning_rate=0.01, max_epochs=10):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = None
        self.cumulative_weights = None
        self.total_updates = 0

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.cumulative_weights = np.zeros(X.shape[1])

        for epoch in range(self.max_epochs):
            for i in range(len(X)):
                prediction = np.sign(np.dot(self.weights, X[i]))
                if prediction == 0:
                    prediction = 1 

                if y[i] != prediction:
                    self.weights += self.learning_rate * y[i] * X[i]
                    self.cumulative_weights += self.weights
                    self.total_updates += 1
                else:
                    self.cumulative_weights += self.weights

        self.weights = self.cumulative_weights / (len(X) * self.max_epochs)

    def predict(self, X):
        return np.sign(np.dot(X, self.weights))

    def calculate_average_error(self, X, y):
        predictions = self.predict(X)
        test_errors = np.sum(predictions != y)
        average_error = test_errors / len(X)
        return average_error

headers = [
    "Variance_Wavelet", 
    "Skewness_Wavelet", 
    "Curtosis_Wavelet", 
    "Entropy", 
    "Label"
]

train_data = pd.read_csv("datasets/bank-note/train.csv", header=None, names=headers)
test_data = pd.read_csv("datasets/bank-note/test.csv", header=None, names=headers)

X_train = train_data.iloc[:, :-1].values 
y_train = train_data.iloc[:, -1].values 

X_test = test_data.iloc[:, :-1].values  
y_test = test_data.iloc[:, -1].values  

y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

model = AveragePerceptron(learning_rate=0.01, max_epochs=10)
model.fit(X_train, y_train)
average_error = model.calculate_average_error(X_test, y_test)

print("Learned weight vector (average):", model.weights)
print("Average prediction error on test dataset:", average_error)