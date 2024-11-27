#!/bin/sh

echo "Result of Primal SVM for 2(a)"
python3 primal1.py

echo "Result of Primal SVM for 2(b)"
python3 primal2.py

echo "Result of dual SVM for 3(a)"
python3 SVMdual.py

echo "Result of Gaussian SVM for 3(b)"
python3 GaussianSVM.py

echo "Result of number of support vectors in SVM for 3(c)"
python3 support_vectors.py

echo "Result of Kernal Perceptron for 3(d)"
python3 kernel_perceptron.py