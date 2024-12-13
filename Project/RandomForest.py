import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

train_df = pd.read_csv('./ml-2024-f/train_final.csv')
test_df = pd.read_csv('./ml-2024-f/test_final.csv')

train_df.replace('?', pd.NA, inplace=True)
test_df.replace('?', pd.NA, inplace=True)

X = train_df.drop('income>50K', axis=1)
y = train_df['income>50K']

categorical_columns = X.select_dtypes(include=['object']).columns
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns

categorical_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
numerical_imputer = SimpleImputer(strategy='median')

X[categorical_columns] = categorical_imputer.fit_transform(X[categorical_columns].astype(str))
X[numerical_columns] = numerical_imputer.fit_transform(X[numerical_columns])

X_test = test_df.drop('ID', axis=1)
X_test[categorical_columns] = categorical_imputer.transform(X_test[categorical_columns].astype(str))
X_test[numerical_columns] = numerical_imputer.transform(X_test[numerical_columns])

X_combined = pd.concat([X, X_test], axis=0)
X_combined = pd.get_dummies(X_combined, columns=categorical_columns, drop_first=True)

X = X_combined.iloc[:len(X)]
X_test = X_combined.iloc[len(X):]

scaler = StandardScaler()
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=10,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_val_pred_proba = rf_model.predict_proba(X_val)[:, 1]
auc_score = roc_auc_score(y_val, y_val_pred_proba)
print(f"Random Forest Validation AUC Score: {auc_score:.3f}")

y_test_pred_proba = rf_model.predict_proba(X_test)[:, 1]

submission_df = pd.DataFrame({
    'ID': test_df['ID'],
    'Prediction': y_test_pred_proba
})

submission_df.to_csv('submission.csv', index=False)
print("Submission saved to 'submission.csv'.")
