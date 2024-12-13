import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel

train_data = pd.read_csv("ml-2024-f/train_final.csv")
test_data = pd.read_csv("ml-2024-f/test_final.csv")

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

train_data, label_encoders = preprocess_data(train_data, is_train=True)
X = train_data.drop(columns=["income>50K"])
y = train_data["income>50K"]

test_data = preprocess_data(test_data, label_encoders, is_train=False)
test_ids = test_data["ID"]
X_test = test_data.drop(columns=["ID"])

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

base_model = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.01, random_state=42)
base_model.fit(X, y)
selector = SelectFromModel(base_model, threshold="median", prefit=True)
X = selector.transform(X)
X_test = selector.transform(X_test)

param_grid = {
    'n_estimators': [2000, 3000],
    'max_depth': [8, 10, 12],
    'learning_rate': [0.002, 0.003],
    'subsample': [0.85, 0.9],
    'colsample_bytree': [0.8, 0.9],
    'reg_alpha': [0.1, 0.2],
    'reg_lambda': [0.3, 0.5],
}

xgb_model = XGBClassifier(scale_pos_weight=1.2, eval_metric="auc", use_label_encoder=False, random_state=42)

grid_search = GridSearchCV(xgb_model, param_grid, scoring="roc_auc", cv=3, verbose=1, n_jobs=-1)
grid_search.fit(X, y)

best_params = grid_search.best_params_
xgb_model = XGBClassifier(**best_params, eval_metric="auc", use_label_encoder=False)

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
auc_scores = []
test_predictions_aggregate = np.zeros(X_test.shape[0])

for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    X_train_kf, X_val_kf = X[train_index], X[val_index]
    y_train_kf, y_val_kf = y[train_index], y[val_index]
    xgb_model.fit(X_train_kf, y_train_kf, eval_set=[(X_val_kf, y_val_kf)], early_stopping_rounds=300, verbose=False)
    val_predictions = xgb_model.predict_proba(X_val_kf)[:, 1]
    auc = roc_auc_score(y_val_kf, val_predictions)
    auc_scores.append(auc)
    test_predictions_aggregate += xgb_model.predict_proba(X_test)[:, 1] / kf.n_splits

print(f"Average AUC Score across Folds: {np.mean(auc_scores):.4f}")

output = pd.DataFrame({"ID": test_ids, "Prediction": test_predictions_aggregate})
output.to_csv("submission.csv", index=False)

print("Predictions saved to submission.csv")
