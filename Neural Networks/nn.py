import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

class NeuralNetworkTorch(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation_function):
        super(NeuralNetworkTorch, self).__init__()
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if activation_function == 'tanh':
                nn.init.xavier_uniform_(layers[-1].weight)
                layers.append(nn.Tanh())
            elif activation_function == 'relu':
                nn.init.kaiming_uniform_(layers[-1].weight, nonlinearity='relu')
                layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

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

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Hyperparameters
depths = [3, 5, 9]
widths = [5, 10, 25, 50, 100]
activation_functions = ['tanh', 'relu']
learning_rate = 1e-3
epochs = 100

results = []

for depth in depths:
    for width in widths:
        for activation_function in activation_functions:
            print(f"\nTraining with depth: {depth}, width: {width}, activation: {activation_function}")

            # Define the network
            hidden_sizes = [width] * depth
            model = NeuralNetworkTorch(X_train.shape[1], hidden_sizes, 1, activation_function)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # Training
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()

                if epoch % 10 == 0:
                    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

            # Evaluate on train and test data
            model.eval()
            with torch.no_grad():
                train_outputs = model(X_train)
                train_error = torch.mean((train_outputs.round() != y_train).float()).item()

                test_outputs = model(X_test)
                test_error = torch.mean((test_outputs.round() != y_test).float()).item()

            results.append({
                'Depth': depth,
                'Width': width,
                'Activation Function': activation_function,
                'Train Error': train_error,
                'Test Error': test_error
            })

            print(f"Train Error: {train_error:.4f} | Test Error: {test_error:.4f}")

# Report results
table = tabulate(results, headers="keys", tablefmt="grid")
print(table)
