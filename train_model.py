import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier


df = pd.read_csv("Customer-Churn.csv")


df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

selected_cols = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "InternetService",
    "PaymentMethod",
    "PaperlessBilling",
    "TechSupport",
    "OnlineSecurity",
    "StreamingMovies",
    "Churn"
]

df_small = df[selected_cols].copy()


df_small["Churn"] = df_small["Churn"].map({"Yes": 1, "No": 0})


X = df_small.drop("Churn", axis=1)
y = df_small["Churn"]


X = pd.get_dummies(X, drop_first=True)

feature_columns = X.columns.tolist()
joblib.dump(feature_columns, "churn_feature_columns.pkl")


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


final_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
    eval_metric="logloss"
)

final_model.fit(X_train, y_train)


y_pred = final_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

cv_scores = cross_val_score(
    final_model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("Cross Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())


joblib.dump(final_model, "churn_model.pkl")

print("\nModel saved successfully.")
print("Feature columns saved successfully.")