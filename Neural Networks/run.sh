#!/bin/sh

echo "Result of back propagation:"
python3 back_propagation.py

echo "Results of Stoghastic Gradient:"
python3 SGD.py

echo "Results of Stochastic Gradient with 0 initialization:"
python3 SGD0.py

echo "Results of Bonus question:"
python3 nn.py