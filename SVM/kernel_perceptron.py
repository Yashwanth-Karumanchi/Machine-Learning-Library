import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score
from tabulate import tabulate
from itertools import product

# Gaussian Kernel
def gaussian_kernel(x1, x2, gamma):
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / gamma)

# Kernel Perceptron Implementation
class KernelPerceptron:
    def __init__(self, gamma, max_iter=100):
        self.gamma = gamma
        self.max_iter = max_iter
        self.alpha = None
        self.support_vectors = None
        self.support_labels = None

    def train(self, X, y):
        n = len(y)
        self.alpha = np.zeros(n)  # Mistake counts for each data point
        self.support_vectors = X  # All points are used as support vectors
        self.support_labels = y

        # Training for max_iter epochs
        for iteration in range(self.max_iter):
            for i in range(n):
                # Compute prediction using kernel
                prediction = np.sign(
                    np.sum(
                        self.alpha * self.support_labels *
                        np.array([gaussian_kernel(X[j], X[i], self.gamma) for j in range(n)])
                    )
                )
                # Update alpha if prediction is incorrect
                if prediction != y[i]:
                    self.alpha[i] += 1

    def predict(self, X):
        predictions = []
        for x in X:
            prediction = np.sign(
                np.sum(
                    self.alpha * self.support_labels *
                    np.array([gaussian_kernel(sv, x, self.gamma) for sv in self.support_vectors])
                )
            )
            predictions.append(prediction)
        return np.array(predictions)

# Kernel SVM Implementation
class KernelSVM:
    def __init__(self, C, gamma):
        self.C = C
        self.gamma = gamma
        self.alpha = None
        self.support_vectors = None
        self.support_labels = None
        self.support_alpha = None
        self.b = 0

    def train(self, X, y):
        n, d = X.shape
        
        # Compute Kernel Matrix
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = gaussian_kernel(X[i], X[j], self.gamma)
        
        # Dual objective function
        def dual_objective(alpha):
            return 0.5 * np.sum((alpha[:, None] * y[:, None]) @ (alpha[None, :] * y[None, :]) * K) - np.sum(alpha)
        
        # Equality constraint: sum(alpha * y) = 0
        constraints = [{'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y)}]

        # Bounds: 0 <= alpha_i <= C
        bounds = [(0, self.C) for _ in range(n)]

        # Initial guess for alpha
        alpha_init = np.zeros(n)

        # Optimize
        result = minimize(dual_objective, alpha_init, bounds=bounds, constraints=constraints, method='SLSQP')
        self.alpha = result.x

        # Identify support vectors
        support_indices = (self.alpha > 1e-5)
        self.support_vectors = X[support_indices]
        self.support_indices = np.where(support_indices)[0]
        self.support_labels = y[support_indices]
        self.support_alpha = self.alpha[support_indices]

        # Compute bias
        self.b = np.mean(
            [self.support_labels[i] - np.sum(
                self.support_alpha * self.support_labels * K[support_indices][:, i]
            ) for i in range(len(self.support_alpha))]
        )

    def predict(self, X):
        predictions = []
        for x in X:
            prediction = np.sum(
                self.support_alpha * self.support_labels *
                [gaussian_kernel(x, sv, self.gamma) for sv in self.support_vectors]
            ) + self.b
            predictions.append(np.sign(prediction))
        return np.array(predictions)

# Main Function
def main():
    # Load the dataset
    train_data = pd.read_csv("datasets/bank-note/train.csv", header=None)
    test_data = pd.read_csv("datasets/bank-note/test.csv", header=None)

    # Convert labels to {1, -1}
    train_data.iloc[:, -1] = train_data.iloc[:, -1].map({1: 1, 0: -1})
    test_data.iloc[:, -1] = test_data.iloc[:, -1].map({1: 1, 0: -1})

    X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
    X_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values

    # Hyperparameters
    gamma_values = [0.1, 0.5, 1, 5, 100]
    C_values = [100 / 873, 500 / 873, 700 / 873]

    # Store results
    perceptron_results = []
    svm_results = []

    # Train Kernel Perceptron
    for gamma in gamma_values:
        print(f"Training Kernel Perceptron with gamma = {gamma}")

        # Train Kernel Perceptron
        kp_model = KernelPerceptron(gamma=gamma)
        kp_model.train(X_train, y_train)

        # Evaluate Kernel Perceptron
        kp_train_preds = kp_model.predict(X_train)
        kp_test_preds = kp_model.predict(X_test)
        kp_train_error = 1 - accuracy_score(y_train, kp_train_preds)
        kp_test_error = 1 - accuracy_score(y_test, kp_test_preds)

        perceptron_results.append({
            'Gamma': gamma,
            'Train Error': kp_train_error,
            'Test Error': kp_test_error
        })

    # Train Kernel SVM
    for C, gamma in product(C_values, gamma_values):
        print(f"Training Kernel SVM with C = {C} and gamma = {gamma}")

        # Train Kernel SVM
        svm_model = KernelSVM(C=C, gamma=gamma)
        svm_model.train(X_train, y_train)

        # Evaluate Kernel SVM
        svm_train_preds = svm_model.predict(X_train)
        svm_test_preds = svm_model.predict(X_test)
        svm_train_error = 1 - accuracy_score(y_train, svm_train_preds)
        svm_test_error = 1 - accuracy_score(y_test, svm_test_preds)

        svm_results.append({
            'C': C,
            'Gamma': gamma,
            'Train Error': svm_train_error,
            'Test Error': svm_test_error
        })

    # Create DataFrames for results
    perceptron_df = pd.DataFrame(perceptron_results)
    svm_df = pd.DataFrame(svm_results)

    # Display results in two separate tables
    print("\nKernel Perceptron Results:")
    print(tabulate(perceptron_df, headers='keys', tablefmt='pretty'))

    print("\nKernel SVM Results:")
    print(tabulate(svm_df, headers='keys', tablefmt='pretty'))

if __name__ == "__main__":
    main()