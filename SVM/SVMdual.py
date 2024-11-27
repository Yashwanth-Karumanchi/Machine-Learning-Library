import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from scipy.optimize import minimize
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# SVM Class Implementation
class SVM:
    def __init__(self, C, gamma0=0.1, a=0.01, max_epochs=100):
        self.C = C
        self.gamma0 = gamma0
        self.a = a
        self.max_epochs = max_epochs
        self.w = None
        self.b = 0
    
    def train_primal(self, X, y):
        """
        Train the SVM using stochastic sub-gradient descent (Primal).
        """
        n, d = X.shape
        self.w = np.zeros(d)  # Initialize weights
        self.b = 0  # Initialize bias
        updates = 0
        
        for epoch in range(self.max_epochs):
            X, y = shuffle(X, y, random_state=epoch)  # Shuffle data at each epoch
            for i in range(n):
                updates += 1
                eta = self.gamma0 / (1 + (self.gamma0 / self.a) * updates)
                margin = y[i] * (np.dot(X[i], self.w) + self.b)
                
                if margin < 1:
                    self.w = (1 - eta) * self.w + eta * self.C * y[i] * X[i]
                    self.b += eta * self.C * y[i]
                else:
                    self.w = (1 - eta) * self.w

    def train_dual(self, X, y):
        """
        Train the SVM using the dual formulation.
        """
        n, d = X.shape

        # Objective function for dual
        def dual_objective(alpha):
            return 0.5 * np.sum((alpha[:, None] * y[:, None] * X).sum(axis=0) ** 2) - np.sum(alpha)

        # Equality constraint: sum(alpha * y) = 0
        constraints = [{'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y)}]

        # Bounds: 0 <= alpha_i <= C
        bounds = [(0, self.C) for _ in range(n)]

        # Initial guess for alpha
        alpha_init = np.zeros(n)

        # Optimize dual objective
        result = minimize(dual_objective, alpha_init, bounds=bounds, constraints=constraints, method='SLSQP')
        alpha_optimal = result.x

        # Compute weights (w) using optimal alpha
        self.w = np.dot((alpha_optimal * y)[:, None].T, X).flatten()

        # Compute bias (b) using support vectors
        support_vectors = (alpha_optimal > 1e-5) & (alpha_optimal < self.C - 1e-5)
        self.b = np.mean(y[support_vectors] - np.dot(X[support_vectors], self.w))

    def predict(self, X):
        """
        Predict the labels for the given input data.
        """
        return np.sign(np.dot(X, self.w) + self.b)

    def evaluate(self, X, y):
        """
        Evaluate the model's error rate.
        """
        predictions = self.predict(X)
        error_rate = np.mean(predictions != y)
        return error_rate

# Main Function to Load Data and Run Experiments
def main():
    # Load the dataset
    train_data = pd.read_csv("datasets/bank-note/train.csv", header=None)
    test_data = pd.read_csv("datasets/bank-note/test.csv", header=None)

    # Convert labels to {1, -1}
    train_data.iloc[:, -1] = train_data.iloc[:, -1].map({1: 1, 0: -1})
    test_data.iloc[:, -1] = test_data.iloc[:, -1].map({1: 1, 0: -1})

    X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
    X_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values
    
    # Hyperparameter C values
    C_values = [100 / 873, 500 / 873, 700 / 873]
    
    for C in C_values:
        print(f"Training with C = {C}")
        
        # Train using primal SVM
        svm_primal = SVM(C)
        svm_primal.train_primal(X_train, y_train)
        train_error_primal = svm_primal.evaluate(X_train, y_train)
        test_error_primal = svm_primal.evaluate(X_test, y_test)

        print("Primal SVM:")
        print(f"  Weights (w): {svm_primal.w}")
        print(f"  Bias (b): {svm_primal.b}")
        print(f"  Training Error: {train_error_primal}")
        print(f"  Testing Error: {test_error_primal}")

        # Train using dual SVM
        svm_dual = SVM(C)
        svm_dual.train_dual(X_train, y_train)
        train_error_dual = svm_dual.evaluate(X_train, y_train)
        test_error_dual = svm_dual.evaluate(X_test, y_test)

        print("Dual SVM:")
        print(f"  Weights (w): {svm_dual.w}")
        print(f"  Bias (b): {svm_dual.b}")
        print(f"  Training Error: {train_error_dual}")
        print(f"  Testing Error: {test_error_dual}")

        # Compare weights, biases, and errors
        weight_difference = np.linalg.norm(svm_primal.w - svm_dual.w)
        bias_difference = np.abs(svm_primal.b - svm_dual.b)
        print("Comparison:")
        print(f"  Weight Difference: {weight_difference}")
        print(f"  Bias Difference: {bias_difference}")
        print("=" * 30)

if __name__ == "__main__":
    main()