import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import roc_auc_score

# Load datasets
train_data = pd.read_csv("ml-2024-f/train_final.csv")
test_data = pd.read_csv("ml-2024-f/test_final.csv")

# Preprocessing function
def preprocess_data(df, label_encoders=None, is_train=True):
    df = df.copy()
    df.replace("?", np.nan, inplace=True)
    df.fillna("Unknown", inplace=True)
    categorical_columns = df.select_dtypes(include=["object"]).columns
    if is_train:
        label_encoders = {}
        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        return df, label_encoders
    else:
        for col in categorical_columns:
            if col in label_encoders:
                le = label_encoders[col]
                df[col] = df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        return df

# Preprocess training and test data
train_data, label_encoders = preprocess_data(train_data, is_train=True)
X = train_data.drop(columns=["income>50K"])
y = train_data["income>50K"]

test_data = preprocess_data(test_data, label_encoders, is_train=False)
test_ids = test_data["ID"]
X_test = test_data.drop(columns=["ID"])

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train Perceptron model
perceptron_model = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
perceptron_model.fit(X_train, y_train)

# Evaluate model on validation data
val_predictions = perceptron_model.decision_function(X_val)
auc_score = roc_auc_score(y_val, val_predictions)
print(f"Validation AUC Score: {auc_score:.4f}")

# Predict on test data
test_predictions = perceptron_model.decision_function(X_test)

# Save predictions to CSV
output = pd.DataFrame({"ID": test_ids, "Prediction": test_predictions})
output.to_csv("submission.csv", index=False)

print("Predictions saved to submission.csv")