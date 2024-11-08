import pandas as pd
import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, max_epochs=10):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = None

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])

        for epoch in range(self.max_epochs):
            for i in range(len(X)):
                prediction = np.sign(np.dot(self.weights, X[i]))
                if prediction == 0:
                    prediction = 1 

                if y[i] != prediction:
                    self.weights += self.learning_rate * y[i] * X[i]

    def calculate_average_error(self, X, y):
        test_errors = 0
        for i in range(len(X)):
            if y[i] * np.dot(self.weights, X[i]) <= 0:
                test_errors += 1
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

model = Perceptron(learning_rate=0.01, max_epochs=10)
model.fit(X_train, y_train)
average_error = model.calculate_average_error(X_test, y_test)

print("Learned weight vector:", model.weights)
print("Average prediction error on test dataset:", average_error)