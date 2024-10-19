#!/bin/sh

echo "Result of batch_gradienet_descent"
python3 gradient_descent.py

echo "Results for stochastic_gradient_descent"
python3 SGD.py

echo "Results for optimalweight"
python3 optimal_weight.py