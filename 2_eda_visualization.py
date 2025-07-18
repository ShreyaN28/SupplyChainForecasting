# 2_eda_visualization.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Step 1: Load the processed data
df = pd.read_csv('processed_data.csv')

# Step 2: Create folder for visualizations
visuals_dir = 'visuals'
os.makedirs(visuals_dir, exist_ok=True)

# Step 3: Plot Distribution of Target Variable
plt.figure(figsize=(8, 6))
sns.histplot(df['HistoricalSales'], kde=True, bins=30, color='skyblue')
plt.title('Distribution of Historical Sales')
plt.xlabel('Number of Products Sold')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(visuals_dir, 'historical_sales_distribution.png'))
plt.close()

# Step 4: Correlation Heatmap for Numeric Features
numeric_df = df.select_dtypes(include=['float64', 'int64'])
corr = numeric_df.corr()

plt.figure(figsize=(14, 12))
sns.heatmap(corr, cmap='coolwarm', annot=False, fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(visuals_dir, 'correlation_heatmap.png'))
plt.close()

# Step 5: Scatter plot of Discount vs Historical Sales (if applicable)
if 'Discount_offered' in df.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df['Discount_offered'], y=df['HistoricalSales'], alpha=0.6)
    plt.title('Discount Offered vs Historical Sales')
    plt.xlabel('Discount Offered')
    plt.ylabel('Number of Products Sold')
    plt.tight_layout()
    plt.savefig(os.path.join(visuals_dir, 'discount_vs_sales.png'))
    plt.close()

# Step 6: Show top 5 features most correlated with Historical Sales
correlation_with_target = corr['HistoricalSales'].drop('HistoricalSales').sort_values(ascending=False)
top_features = correlation_with_target.head(5)
print("\n🔍 Top 5 features most positively correlated with Historical Sales:")
print(top_features)

print(f"\n✅ Visualizations saved in '{visuals_dir}' folder.")
