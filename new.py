import streamlit as st
import pandas as pd
import requests
import xml.etree.ElementTree as ET
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import re

# 📌 Function to analyze dataset
def analyze_dataset(file):
    df = pd.read_csv(file)
    
    dataset_info = {
        "Shape": df.shape,
        "Columns": df.columns.tolist(),
        "Missing Values": df.isnull().sum().to_dict(),
        "Data Types": df.dtypes.astype(str).to_dict(),
        "Summary Statistics": df.describe().to_dict()
    }
    
    return df, dataset_info

# 📌 Function to detect outliers using IQR
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

# 📌 Function to recommend models based on dataset
def recommend_model(df):
    target_var = df.columns[-1]  # Last column assumed as target

    if df[target_var].dtype in ["int64", "float64"]:
        return "Regression Model Recommended: Random Forest Regressor, XGBoost, Neural Networks"
    else:
        return "Classification Model Recommended: Random Forest, SVM, Logistic Regression"

# 📌 Function to generate a search query from dataset columns
def generate_search_query(df):
    columns = " ".join(df.columns)  # Join all column names into a single string
    words = re.findall(r'\b\w+\b', columns)  # Extract words
    keywords = list(set(words))[:5]  # Pick the first 5 unique words
    return " ".join(keywords) if keywords else "machine learning dataset"

# 📌 Function to fetch research papers from arXiv
def fetch_papers(query):
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=5"
    response = requests.get(url)
    
    if response.status_code == 200:
        root = ET.fromstring(response.text)
        papers = []
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            title = entry.find("{http://www.w3.org/2005/Atom}title").text
            link = entry.find("{http://www.w3.org/2005/Atom}id").text
            papers.append(f"📄 [{title}]({link})")
        return papers if papers else ["No relevant papers found."]
    return ["Error fetching papers."]

# 📌 Function to perform K-Means clustering
def plot_clusters(df):
    numeric_data = df.select_dtypes(include=['number']).dropna()
    if numeric_data.shape[1] > 1:  
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(scaled_data)

        fig, ax = plt.subplots()
        sns.scatterplot(x=numeric_data.iloc[:, 0], y=numeric_data.iloc[:, 1], hue=df['Cluster'], palette="viridis", ax=ax)
        ax.set_title("K-Means Clustering")
        return fig
    else:
        return None

# 📌 Streamlit UI
st.title("🔍 AI-Powered Dataset Analyzer & Model Recommender")

uploaded_file = st.file_uploader("📂 Upload Your Dataset (CSV)", type=["csv"])
if uploaded_file:
    st.success("✅ File Uploaded Successfully!")
    
    df, dataset_info = analyze_dataset(uploaded_file)
    st.subheader("📊 Dataset Overview")
    st.json(dataset_info)

    # 🎯 Target Variable Distribution
    target_var = df.columns[-1]
    if target_var in df.select_dtypes(include=['number']).columns:
        st.subheader(f"🎯 Distribution of Target Variable: {target_var}")
        fig, ax = plt.subplots()
        sns.histplot(df[target_var], kde=True, bins=30, ax=ax)
        st.pyplot(fig)

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

        st.write(f"🔴 Outliers in {selected_feature}: {len(outliers)}")
        st.dataframe(outliers if not outliers.empty else "No significant outliers detected.")

    # 📌 Bar Charts for Categorical Data
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    st.subheader("📊 Bar Chart for Categorical Features")
    
    if len(categorical_cols) > 0:
        selected_category = st.selectbox("📌 Select Categorical Feature", categorical_cols)
        fig, ax = plt.subplots()
        sns.countplot(x=df[selected_category], ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # 📌 Scatter Plot for Feature Relationships
    st.subheader("📌 Scatter Plot for Feature Relationships")
    if len(numeric_cols) > 1:
        x_feature = st.selectbox("Select X-axis feature", numeric_cols, index=0)
        y_feature = st.selectbox("Select Y-axis feature", numeric_cols, index=1)
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[x_feature], y=df[y_feature], ax=ax)
        st.pyplot(fig)

    # 📌 Line Graph for Trend Analysis
    st.subheader("📌 Line Graph for Trend Analysis")
    if len(numeric_cols) > 1:
        line_feature = st.selectbox("Select Feature for Line Graph", numeric_cols)
        fig, ax = plt.subplots()
        sns.lineplot(data=df, x=df.index, y=line_feature, ax=ax)
        st.pyplot(fig)

    # 📌 Clustering Visualization
    st.subheader("📌 K-Means Clustering")
    cluster_fig = plot_clusters(df)
    if cluster_fig:
        st.pyplot(cluster_fig)
    else:
        st.write("Not enough numerical features for clustering.")

    # 📌 Violin Plot for Data Distribution
    st.subheader("📌 Violin Plot for Data Distribution")
    if len(numeric_cols) > 0:
        violin_feature = st.selectbox("Select Feature for Violin Plot", numeric_cols)
        fig, ax = plt.subplots()
        sns.violinplot(x=df[violin_feature], ax=ax)
        st.pyplot(fig)

    # 🤖 Recommend Model
    recommended_model = recommend_model(df)
    st.subheader("🤖 Recommended Model")
    st.write(recommended_model)

    # 📚 Fetch Research Papers Related to Dataset
    dataset_query = generate_search_query(df)
    papers = fetch_papers(dataset_query)

    st.subheader("📚 Research Papers Related to Dataset")
    for paper in papers:
        st.markdown(paper)