# 4_model_evaluation.py

import pandas as pd
import joblib
import numpy as np

# === Load the saved model, scaler, and encoders ===
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("label_encoders.pkl")

# === Load original data and simulate new unseen data ===
data = pd.read_csv("cleaned_supply_chain_data.csv")

# Drop the target column to simulate new data
data_new = data.drop(columns=["Order quantities"])
data_new.to_csv("new_data_for_prediction.csv", index=False)
print("✅ Saved 'new_data_for_prediction.csv' without target column for testing predictions.\n")

# === Preprocess new data ===

# Encode categorical columns
for col, encoder in encoders.items():
    if col in data_new.columns:
        data_new[col] = encoder.transform(data_new[col])

# Scale numerical features
data_new_scaled = scaler.transform(data_new)

# === Make predictions ===
predictions = model.predict(data_new_scaled)

# === Combine predictions with original data ===
results_df = data_new.copy()
results_df["Predicted Order Quantities"] = predictions

# Save predictions to CSV
results_df.to_csv("predicted_order_quantities.csv", index=False)
print("✅ Predictions saved to 'predicted_order_quantities.csv'.\n")

# Display sample predictions
print("📊 Sample Predictions:")
print(results_df[["Predicted Order Quantities"]].head())
