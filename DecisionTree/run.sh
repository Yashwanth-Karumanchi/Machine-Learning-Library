#!/bin/sh

if python3 -c "import tabulate" >/dev/null 2>&1; then
    echo "Tabulate library is already installed."
else
    echo "Tabulate library is not installed. Installing now..."
    pip3 install tabulate
fi

echo "Training and Test errors for car dataset"
python3 decisionTree_car.py

echo "Training and Test errors for bank dataset"
python3 decisionTree_bank.py

echo "Training and Test errors for bank dataset with unknown replaced with most Common Values"
python3 decisionTree_bank_unknown.p