import pandas as pd
import joblib

# Step 1: Load cleaned data
df = pd.read_csv("cleaned_supply_chain_data.csv")

# Step 2: Rename the target column if needed
df.rename(columns={'Number of products sold': 'HistoricalSales'}, inplace=True)

# Step 3: Separate features and target
X = df.drop('HistoricalSales', axis=1)
y = df['HistoricalSales']

# Step 4: One-hot encode categorical variables
X_encoded = pd.get_dummies(X)

# Step 5: Save column structure for future consistency
joblib.dump(X_encoded.columns.tolist(), 'X_columns.pkl')

# Step 6: Save processed data
X_encoded['HistoricalSales'] = y
X_encoded.to_csv("processed_data.csv", index=False)

print("✅ Data preprocessed and saved to 'processed_data.csv'")
print(f"📊 Features shape after encoding: {X_encoded.shape}")
