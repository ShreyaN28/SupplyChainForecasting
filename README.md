#  Supply Chain Forecasting using Machine Learning

This project focuses on building a predictive model to forecast **Order Quantities** in a supply chain using machine learning techniques. It covers the entire data science workflow – from data cleaning and exploratory data analysis to model building, evaluation, and deployment using **Streamlit**.

---

##  Project Objectives

- To clean and preprocess raw supply chain data.
- To explore and visualize important trends in the dataset.
- To train and evaluate multiple regression models.
- To deploy the best-performing model in an interactive web app using Streamlit.

---
##  Key Sections

### 1. Data Cleaning (`1_data_cleaning.py`)
- Removed missing/null values.
- Renamed columns for clarity.
- Encoded categorical variables using LabelEncoder.
- Saved cleaned dataset for further analysis.

### 2. Exploratory Data Analysis & Visualization (`2_eda_visualization.py`)
- Generated correlation heatmaps.
- Visualized top contributing factors to order quantity.
- Plotted histograms and boxplots for outlier analysis.

### 3. Model Training (`3_model_training.py`)
- Tested multiple regression models:
  - Linear Regression
  - Random Forest Regressor
  - KNN
  - Gradient Boosting Regressor
- Saved the best-performing model using `joblib`.

### 4. Model Evaluation & Prediction (`4_model_evaluation.py`)
- Simulated prediction on unseen data.
- Scaled and encoded features using saved `scaler.pkl` and `label_encoders.pkl`.
- Predicted values and exported them to `predicted_order_quantities.csv`.

---

## Streamlit Dashboard (`app.py`)

Built an interactive web app using Streamlit to:
- Upload and preprocess supply chain data.
- Visualize trends.
- Show predictions from the trained model.
- Display summary statistics and downloadable results.
