import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score
from itertools import product
from tabulate import tabulate  # For clean output formatting

# Gaussian Kernel
def gaussian_kernel(x1, x2, gamma):
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / gamma)

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
        self.support_labels = y[support_indices]
        self.support_alpha = self.alpha[support_indices]

        # Compute bias
        self.b = np.mean(
            [self.support_labels[i] - np.sum(
                self.support_alpha * self.support_labels * K[support_indices][:, i]
            ) for i in range(len(self.support_alpha))]
        )

    def predict(self, X):
        # Predict using kernel and support vectors
        y_pred = []
        for x in X:
            prediction = np.sum(
                self.support_alpha * self.support_labels *
                [gaussian_kernel(x, sv, self.gamma) for sv in self.support_vectors]
            ) + self.b
            y_pred.append(np.sign(prediction))
        return np.array(y_pred)

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
    C_values = [100 / 873, 500 / 873, 700 / 873]
    gamma_values = [0.1, 0.5, 1, 5, 100]

    # Store results
    results = []

    for C, gamma in product(C_values, gamma_values):
        print(f"Training with C = {C}, gamma = {gamma}")

        # Train the Kernel SVM
        model = KernelSVM(C=C, gamma=gamma)
        model.train(X_train, y_train)

        # Evaluate on train and test data
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        train_error = 1 - accuracy_score(y_train, train_preds)
        test_error = 1 - accuracy_score(y_test, test_preds)

        results.append({
            'C': C,
            'Gamma': gamma,
            'Train Error': train_error,
            'Test Error': test_error
        })

    # Create DataFrame of results
    results_df = pd.DataFrame(results)

    # Display the results in a tabular format
    print(tabulate(results_df, headers='keys', tablefmt='pretty'))

if __name__ == "__main__":
    main()
