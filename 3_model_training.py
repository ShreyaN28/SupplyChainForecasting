# 3_model_training.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load cleaned data
data = pd.read_csv("cleaned_supply_chain_data.csv")

# Target column to predict
target_column = 'Order quantities'

# Split features and target
X = data.drop(columns=[target_column])
y = data[target_column]

# Encode categorical features
categorical_cols = X.select_dtypes(include='object').columns
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "KNN": KNeighborsRegressor(n_neighbors=5)
}

# Train, evaluate, and compare models
best_model = None
best_model_name = ""
best_r2 = -np.inf

print("\nModel Evaluation Results:")
print(f"{'Model':<20} {'R² Score':<10} {'MAE':<10}")
print("-" * 42)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"{name:<20} {r2:<10.4f} {mae:<10.2f}")

    if r2 > best_r2:
        best_model = model
        best_model_name = name
        best_r2 = r2

# Save best model, scaler, encoders
joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoders, "label_encoders.pkl")

print(f"\n✅ Best model: {best_model_name} (R² = {best_r2:.4f}) saved as 'best_model.pkl'")
