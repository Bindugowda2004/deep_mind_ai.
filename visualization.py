import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# 📌 Function to analyze dataset
def analyze_dataset(file):
    df = pd.read_csv(file)
    return df

# 📌 Function to detect outliers using IQR
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

# 📌 Function to suggest model based on dataset
def suggest_model(df):
    target_var = df.columns[-1]  # Assuming last column as target

    if df[target_var].dtype in ["int64", "float64"]:
        model_info = {
            "Model Type": "Regression",
            "Suggested Model": "XGBoost Regressor",
            "Why This Model": "XGBoost is an ensemble learning method that combines multiple decision trees to make predictions, reducing overfitting.",
            "Why It Performs Better": "It uses gradient boosting, which optimizes loss functions efficiently and performs well even with missing values.",
            "Key Advantages": [
                "Handles missing values effectively.",
                "Provides high accuracy compared to linear models.",
                "Fast training speed due to parallel processing."
            ],
            "Use Case": "Predicting continuous values such as sales, prices, or customer spend."
        }
    else:
        model_info = {
            "Model Type": "Classification",
            "Suggested Model": "Random Forest Classifier",
            "Why This Model": "Random Forest is an ensemble of decision trees, making it more robust to overfitting and noise in data.",
            "Why It Performs Better": "It can handle both categorical and numerical features effectively and provides better accuracy than single decision trees.",
            "Key Advantages": [
                "Reduces overfitting by averaging multiple trees.",
                "Can handle large datasets efficiently.",
                "Works well with imbalanced data."
            ],
            "Use Case": "Classifying data into categories, such as spam detection, customer segmentation, or fraud detection."
        }
    
    return model_info

# 📌 Function to visualize PCA
def plot_pca(df):
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return None  # PCA requires at least 2 numerical features
    
    # Handle missing values by replacing NaN with column mean
    imputer = SimpleImputer(strategy="mean")
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    # Check if too many missing values were present
    missing_values_count = df.isnull().sum().sum()
    if missing_values_count > 0:
        st.warning(f"⚠ Warning: {missing_values_count} missing values were replaced with column means.")

    # Standardize the data
    scaled_data = StandardScaler().fit_transform(df[numeric_cols])
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    
    fig, ax = plt.subplots()
    ax.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("PCA Visualization")
    
    return fig

# 📌 Streamlit UI
st.title("🔍 AI-Powered Dataset Analyzer & Suggested Model")

uploaded_file = st.file_uploader("📂 Upload Your Dataset (CSV)", type=["csv"])
if uploaded_file:
    st.success("✅ File Uploaded Successfully!")
    
    df = analyze_dataset(uploaded_file)

    # 📈 Correlation Heatmap
    st.subheader("📊 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)

    # 📉 Boxplots for Outliers
    st.subheader("📦 Boxplot & Outliers Detection")
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_cols) > 0:
        selected_feature = st.selectbox("📌 Select Feature for Boxplot", numeric_cols)
        outliers = detect_outliers(df, selected_feature)

        fig, ax = plt.subplots()
        sns.boxplot(x=df[selected_feature], ax=ax, flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 8})
        st.pyplot(fig)

        if not outliers.empty:
            st.write(f"🔴 Outliers detected in {selected_feature}: {len(outliers)}")
            st.dataframe(outliers)
        else:
            st.write("✅ No significant outliers detected.")

    # 📌 PCA Graph
    st.subheader("📌 PCA Visualization")
    pca_fig = plot_pca(df)
    if pca_fig:
        st.pyplot(pca_fig)
    else:
        st.write("❌ PCA requires at least two numerical features.")

    # 🤖 Suggested Model
    suggested_model = suggest_model(df)
    st.subheader("🤖 Suggested Model")
    st.write(f"📌 Model Type: {suggested_model['Model Type']}")
    st.write(f"📌 Suggested Model: {suggested_model['Suggested Model']}")
    st.write(f"📌 Why This Model? {suggested_model['Why This Model']}")
    st.write(f"📌 Why It Performs Better? {suggested_model['Why It Performs Better']}")
    st.write("📌 Key Advantages:")
    for adv in suggested_model["Key Advantages"]:
        st.write(f"- {adv}")
    st.write(f"📌 Use Case: {suggested_model['Use Case']}")