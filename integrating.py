import os
import streamlit as st
import pandas as pd
import google.generativeai as genai
import requests
import xml.etree.ElementTree as ET
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Set API Key
# genai.configure(api_key="_")  # Replace with actual API key
# model = genai.GenerativeModel("gemini-1.5-pro-002")

def analyze_dataset(file):
    df = pd.read_csv(file)
    dataset_name = os.path.splitext(file.name)[0]
    dataset_info = {
        "Dataset Name": dataset_name,
        "Shape": df.shape,
        "Columns": df.columns.tolist(),
        "Missing Values": df.isnull().sum().to_dict(),
        "Data Types": df.dtypes.astype(str).to_dict(),
        "Target Variable": df.columns[-1] if len(df.columns) > 1 else "Unknown"
    }
    return df, dataset_info

def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

def recommend_model(df):
    target_var = df.columns[-1]
    if target_var in df.select_dtypes(include=["int64", "float64"]).columns:
        return "Regression Model Recommended: Random Forest Regressor, XGBoost, Neural Networks"
    else:
        return "Classification Model Recommended: Random Forest, SVM, Logistic Regression"

def fetch_papers(query):
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=5"
    response = requests.get(url)
    if response.status_code == 200:
        root = ET.fromstring(response.text)
        return [f"📄 [{entry.find('{http://www.w3.org/2005/Atom}title').text}]({entry.find('{http://www.w3.org/2005/Atom}id').text})" for entry in root.findall("{http://www.w3.org/2005/Atom}entry")] or ["No relevant papers found."]
    return ["Error fetching papers."]

def get_ai_insights(dataset_info):
    prompt = f"""
    You are an expert in machine learning and data science. Based on the dataset details below, provide a detailed report:
    
    *Dataset Summary*:
    - Shape: {dataset_info["Shape"]}
    - Columns: {dataset_info["Columns"]}
    - Missing Values: {dataset_info["Missing Values"]}
    - Data Types: {dataset_info["Data Types"]}
    - Target Variable: {dataset_info["Target Variable"]}
    
    *Tasks*:
    1. Describe the dataset in one line.
    2. Suggest the best ML models for analysis.
    3. Identify key preprocessing steps.
    4. Recommend feature engineering techniques.
    5. Outline EDA steps.
    6. Identify attribute interdependencies and non-dependent values.
    7. Perform PCA analysis on the dataset.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error fetching insights: {str(e)}"

# Streamlit UI
st.title("📊 AI-Powered Dataset Analyzer (Gemini)")

uploaded_file = st.file_uploader("📂 Upload Your Dataset (CSV)", type=["csv"])
if uploaded_file:
    df, dataset_info = analyze_dataset(uploaded_file)
    st.success("✅ File Uploaded Successfully!")
    st.subheader("📈 Dataset Overview")
    st.json(dataset_info)
    
    st.subheader("🧠 AI-Powered Insights")
    with st.spinner("Analyzing dataset..."):
        insights = get_ai_insights(dataset_info)
    st.markdown(insights)
    
    st.subheader("🤖 Recommended Model")
    recommended_model = recommend_model(df)
    st.write(recommended_model)
    
    search_query = recommended_model.split(":")[1].split(",")[0].strip() if ":" in recommended_model else "Machine Learning"
    papers = fetch_papers(search_query)
    st.subheader("📚 Research Papers Using Similar Models")
    for paper in papers:
        st.markdown(paper)
    
    st.subheader("📊 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)
    
    st.subheader("📦 Boxplot & Outliers Detection")
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        selected_feature = st.selectbox("📌 Select Feature for Boxplot", numeric_cols)
        outliers = detect_outliers(df, selected_feature)
        fig, ax = plt.subplots()
        sns.boxplot(x=df[selected_feature], ax=ax, flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 8})
        st.pyplot(fig)
        if not outliers.empty:
            st.write(f"🔴 *Outliers in {selected_feature}:* {len(outliers)}")
            st.dataframe(outliers)
        else:
            st.write("No significant outliers detected.")
    
    if len(numeric_cols) > 1:
        st.subheader("📉 PCA for Dimensionality Reduction")
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df[numeric_cols])
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(df_scaled)
        pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        fig, ax = plt.subplots()
        sns.scatterplot(x=pca_df['PC1'], y=pca_df['PC2'], ax=ax)
        st.pyplot(fig)
