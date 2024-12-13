import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

df = pd.read_csv('./ml-2024-f/train_final.csv')
X = df.drop('income>50K', axis=1)
y = df['income>50K']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_columns = X_train.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    label_encoders[col] = le

numerical_columns = X_train.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

logistic_model = LogisticRegression(solver='liblinear', random_state=42)
logistic_model.fit(X_train, y_train)

y_pred_proba = logistic_model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"Logistic Regression AUC Score: {auc_score:.4f}")

df_test = pd.read_csv('./ml-2024-f/test_final.csv')
X_new = df_test.drop('ID', axis=1)

for col in categorical_columns:
    X_new[col] = X_new[col].apply(lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1)

X_new[numerical_columns] = scaler.transform(X_new[numerical_columns])
y_new_pred_proba = logistic_model.predict_proba(X_new)[:, 1]

submission_df = pd.DataFrame({
    'ID': df_test['ID'],
    'Prediction': y_new_pred_proba
})

submission_df.to_csv('submission.csv', index=False)
print("Kaggle submission saved to 'submission.csv'.")
