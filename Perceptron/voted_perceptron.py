import pandas as pd
import numpy as np

class VotedPerceptron:
    def __init__(self, learning_rate=0.01, max_epochs=10):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weight_vectors = []
        self.counts = []
        self.epoch_misclassifications = []

    def fit(self, X, y):
        weights = np.zeros(X.shape[1])
        n_samples = len(X)

        for epoch in range(self.max_epochs):
            mistakes = 0

            for i in range(n_samples):
                prediction = np.sign(np.dot(weights, X[i]))
                if prediction == 0:
                    prediction = -1

                if y[i] != prediction:
                    weights += self.learning_rate * y[i] * X[i]
                    mistakes += 1

            self.epoch_misclassifications.append(mistakes)

            # Store only the weight vector and count corresponding to n_samples - mistakes
            self.weight_vectors.append(weights.copy())
            self.counts.append(n_samples - mistakes)

    def predict(self, X):
        predictions = []
        for x in X:
            vote_sum = 0
            for weights, count in zip(self.weight_vectors, self.counts):
                vote = np.sign(np.dot(weights, x))
                vote_sum += count * vote
            predictions.append(np.sign(vote_sum))
        return np.array(predictions)

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

model = VotedPerceptron(learning_rate=0.01, max_epochs=10)
model.fit(X_train, y_train)

average_error = model.calculate_average_error(X_test, y_test)

print("\nWeight vectors and their correct counts:")
for i, (w, count) in enumerate(zip(model.weight_vectors, model.counts)):
    print(f"Weight Vector {i + 1}: {w}, Correct Count: {count}")

print(f"\nAverage prediction error on test dataset: {average_error:.2f}")