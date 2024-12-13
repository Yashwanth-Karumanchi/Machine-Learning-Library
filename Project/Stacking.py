import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

df = pd.read_csv('./ml-2024-f/train_final.csv')
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
numerical_imputer = SimpleImputer(strategy='median')
df[numerical_columns] = numerical_imputer.fit_transform(df[numerical_columns])
categorical_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
X = df.drop('income>50K', axis=1)
y = df['income>50K'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
xrt_model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.6,
    colsample_bytree=0.5,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xrt_model.fit(X_train, y_train)
y_pred_proba = xrt_model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"XGBoost (XRT) AUC Score: {auc_score:.3f}")

df_test = pd.read_csv('./ml-2024-f/test_final.csv')
X_new = df_test.drop('ID', axis=1)
numerical_columns = [col for col in X_new.select_dtypes(include=['int64', 'float64']).columns]
numerical_imputer = SimpleImputer(strategy='median')
X_new[numerical_columns] = numerical_imputer.fit_transform(X_new[numerical_columns])
X_new = pd.get_dummies(X_new, drop_first=True)
scaler = StandardScaler()
X_new[numerical_columns] = scaler.fit_transform(X_new[numerical_columns])
X_new = X_new.reindex(columns=X_train.columns, fill_value=0)
y_new_pred_proba = xrt_model.predict_proba(X_new)[:, 1]
submission_df = pd.DataFrame({
    'ID': df_test['ID'],
    'Prediction': y_new_pred_proba
})
submission_df.to_csv('submission.csv', index=False, header=True, sep='\t')
print("Predictions saved to 'submission.csv'.")
