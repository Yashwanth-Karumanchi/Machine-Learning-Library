#!/bin/sh

echo "Result of Logistic Regression:"
python3 LogisticRegression.py

echo "Results of Random Forest:"
python3 RandomForest.py

echo "Results of XG Boost:"
python3 XGB.py

echo "Results of Stacking:"
python3 Stacking.py

echo "Results of Perceptron:"
python3 perceptron.py

echo "Results of Neural Network:"
python3 NeuralNetwork.py