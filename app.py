# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
@st.cache_data
def load_data():
    original_data = pd.read_csv("cleaned_supply_chain_data.csv")
    prediction_data = pd.read_csv("predicted_order_quantities.csv")
    return original_data, prediction_data

# Load files
original_data, prediction_data = load_data()

# App layout
st.set_page_config(layout="wide", page_title="Supply Chain Forecasting Dashboard")
st.title("📦 Supply Chain Demand Forecasting - Final Dashboard")

st.sidebar.title("🔍 Navigation")
section = st.sidebar.radio("Go to:", ["Dataset Overview", "Dashboard", "Model Performance", "Actual vs Predicted"])

st.sidebar.markdown("#### 📋 Dataset Columns")
st.sidebar.write(list(original_data.columns))

# --- 1. Dataset Overview ---
if section == "Dataset Overview":
    st.header("📄 Dataset Overview")
    st.subheader("First 10 Rows of Cleaned Data")
    st.dataframe(original_data.head(10), use_container_width=True)

    st.subheader("🔢 Data Summary")
    st.write(original_data.describe())

# --- 2. Dashboard: Visual Insights ---
elif section == "Dashboard":
    st.header("📊 Visual Insights from Dataset")

    # --- Top 10 categories based on available categorical column ---
    categorical_cols = original_data.select_dtypes(include='object').columns.tolist()

    if categorical_cols:
        selected_col = st.selectbox("Select a categorical column to view Top 10", categorical_cols)
        top_grouped = original_data.groupby(selected_col)["Order quantities"].sum().sort_values(ascending=False).head(10)
        st.subheader(f"1️⃣ Top 10 {selected_col} by Order Quantities")
        st.bar_chart(top_grouped)
    else:
        st.warning("No categorical column available to group top 10 by.")

    # --- Correlation Heatmap ---
    st.subheader("2️⃣ Correlation Heatmap")
    corr = original_data.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # --- Line chart for Order Quantities ---
    st.subheader("3️⃣ Order Quantities Trend (First 100 rows)")
    st.line_chart(original_data["Order quantities"].head(100))

# --- 3. Model Performance ---
elif section == "Model Performance":
    st.header("📈 Model Performance Metrics")

    st.markdown("""### 🔍 Evaluation Summary  
We tested different models and selected the one with the **best performance based on R² Score** and **MAE (Mean Absolute Error)**.
The best model was: `K-Nearest Neighbors (KNN)`  
""")

    st.code("""
Model                R² Score     MAE
-------------------------------------
Linear Regression    -0.2639    23.57
Decision Tree        -0.5204    26.07
Random Forest        -0.5479    27.39
KNN                  -0.1617    20.76   <- Best
""")

# --- 4. Actual vs Predicted ---
elif section == "Actual vs Predicted":
    st.header("🔄 Actual vs Predicted Order Quantities")

    # Load prediction results
    if "Predicted Order Quantities" in prediction_data.columns:
        num_rows = st.slider("Select number of rows to compare", 10, len(prediction_data), 50)
        prediction_data_subset = prediction_data.head(num_rows)

        st.subheader("📊 Comparison Table")
        st.dataframe(prediction_data_subset[["Predicted Order Quantities"]], use_container_width=True)

        st.subheader("📉 Visualization: Predicted Order Quantities")
        st.line_chart(prediction_data_subset["Predicted Order Quantities"])

    else:
        st.error("Prediction file doesn't contain 'Predicted Order Quantities'. Please check the data.")
