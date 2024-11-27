import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class SVM:
    def __init__(self, C, gamma_0, max_epochs):
        """
        Initializes the SVM model with the given hyperparameters.

        Args:
            C (float): Regularization parameter.
            gamma_0 (float): Initial learning rate.
            max_epochs (int): Number of training epochs.
        """
        self.C = C
        self.gamma_0 = gamma_0
        self.max_epochs = max_epochs
        self.weights = None
        self.bias = None

    def _learning_rate(self, t):
        """Calculates the decayed learning rate as γ_t = γ_0 / (1 + t)."""
        return self.gamma_0 / (1 + t)

    def fit(self, X, y):
        """
        Train the SVM using stochastic sub-gradient descent with learning rate schedule.

        Args:
            X (ndarray): Training feature matrix.
            y (ndarray): Training labels.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        objective_values = []

        # Training loop
        t = 0  # Global iteration counter
        for epoch in range(self.max_epochs):
            indices = np.random.permutation(n_samples)  # Shuffle indices at the start of each epoch
            for i in indices:
                lr = self._learning_rate(t)
                t += 1
                condition = y[i] * (np.dot(self.weights, X[i]) + self.bias)
                if condition <= 1:
                    self.weights = (1 - lr) * self.weights + lr * self.C * y[i] * X[i]
                    self.bias += lr * self.C * y[i]
                else:
                    self.weights = (1 - lr) * self.weights

            # Objective function for diagnostics
            margin_losses = np.maximum(0, 1 - y * (np.dot(X, self.weights) + self.bias))
            obj_value = 0.5 * np.dot(self.weights, self.weights) + self.C * np.sum(margin_losses)
            objective_values.append(obj_value)

        # Plot convergence of the objective function
        plt.figure(figsize=(8, 5))
        plt.plot(objective_values, label="Objective Function")
        plt.xlabel("Epochs")
        plt.ylabel("Objective Function Value")
        plt.title("Convergence of Objective Function")
        plt.legend()
        plt.grid()
        plt.show()

    def predict(self, X):
        """Predicts labels for the input features."""
        return np.sign(np.dot(X, self.weights) + self.bias)

    def evaluate(self, X, y):
        """
        Evaluates the model on given data.

        Args:
            X (ndarray): Input feature matrix.
            y (ndarray): True labels.

        Returns:
            float: Error rate.
        """
        predictions = self.predict(X)
        error_rate = np.mean(predictions != y)
        return error_rate


def preprocess_data(file_path):
    """
    Preprocess the dataset.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        tuple: Feature matrix (X) and labels (y).
    """
    data = pd.read_csv(file_path, header=None)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    y = np.where(y == 0, -1, 1)  # Map labels to {1, -1}
    
    # Normalize features to zero mean and unit variance
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    return X, y


def main():
    # Paths to data files
    train_file = "datasets/bank-note/train.csv"
    test_file = "datasets/bank-note/test.csv"

    # Load and preprocess data
    X_train, y_train = preprocess_data(train_file)
    X_test, y_test = preprocess_data(test_file)

    # Hyperparameters
    Cs = [100 / 873, 500 / 873, 700 / 873]
    gamma_0 = 0.01
    max_epochs = 100

    # Train and evaluate the SVM for different values of C
    for C in Cs:
        print(f"\nTraining SVM with C = {C}")
        model = SVM(C, gamma_0, max_epochs)
        model.fit(X_train, y_train)

        train_error = model.evaluate(X_train, y_train)
        test_error = model.evaluate(X_test, y_test)

        print(f"Training error: {train_error:.4f}")
        print(f"Test error: {test_error:.4f}")


if __name__ == "__main__":
    main()
