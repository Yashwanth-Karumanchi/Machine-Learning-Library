import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

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

# Define the ANN model
def build_model(input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['AUC'])
    return model

# Train and evaluate using K-Fold cross-validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
auc_scores = []
test_predictions_aggregate = np.zeros(X_test.shape[0])

for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    print(f"Training Fold {fold + 1}...")
    X_train_kf, X_val_kf = X[train_index], X[val_index]
    y_train_kf, y_val_kf = y[train_index], y[val_index]
    
    model = build_model(input_dim=X.shape[1])
    model.fit(X_train_kf, y_train_kf, epochs=50, batch_size=32, validation_data=(X_val_kf, y_val_kf), verbose=0)
    
    val_predictions = model.predict(X_val_kf).flatten()
    auc = roc_auc_score(y_val_kf, val_predictions)
    auc_scores.append(auc)
    print(f"Fold {fold + 1} AUC: {auc:.4f}")
    
    test_predictions_aggregate += model.predict(X_test).flatten() / kf.n_splits

# Final Average AUC Score
print(f"Average AUC Score across Folds: {np.mean(auc_scores):.4f}")

# Save predictions to CSV
output = pd.DataFrame({"ID": test_ids, "Prediction": test_predictions_aggregate})
output.to_csv("submission.csv", index=False)

print("Predictions saved to submission.csv")