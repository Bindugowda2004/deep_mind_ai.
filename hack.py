import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from scipy import stats
import io

# Page Configuration
st.set_page_config(page_title="Dataset Analysis", page_icon="📊", layout="wide")

# Session State Initialization
if "page" not in st.session_state:
    st.session_state["page"] = "signup"

# Page Navigation Function
def go_to_page(page_name):
    st.session_state["page"] = page_name
    st.rerun()

# --- PAGE 1: SIGNUP PAGE ---
if st.session_state["page"] == "signup":
    st.title("Welcome! Sign Up to Continue")

    username = st.text_input("Enter your username")
    email = st.text_input("Enter your email")

    if st.button("Next"):
        if username and email:
            st.session_state["username"] = username
            st.session_state["email"] = email
            go_to_page("upload")
        else:
            st.error("Please fill in both fields.")

# --- PAGE 2: DATASET UPLOAD PAGE ---
elif st.session_state["page"] == "upload":
    st.title("Upload Your Dataset 📤")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    dataset_purpose = st.text_area("What is the purpose of your dataset?")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state["df"] = df  # Save dataset to session state

        if st.button("Next"):
            go_to_page("description")

# --- PAGE 3: DATA DESCRIPTION & PREPROCESSING ---
elif st.session_state["page"] == "description":
    st.title("Dataset Summary and Preprocessing 🔍")

    if "df" in st.session_state:
        df = st.session_state["df"]

        # Display Dataset Summary
        st.subheader("1️⃣ Dataset Overview")
        st.write(df.head())

        # Show basic statistics
        st.subheader("2️⃣ Summary Statistics")
        st.write(df.describe())

        # Check Missing Values
        st.subheader("3️⃣ Handling Missing Values")
        missing_values = df.isnull().sum()
        st.write(missing_values[missing_values > 0])

        if st.button("Fill Missing Values with Mean/Mode"):
            for col in df.select_dtypes(include=["number"]).columns:
                df[col].fillna(df[col].mean(), inplace=True)
            for col in df.select_dtypes(include=["object"]).columns:
                df[col].fillna(df[col].mode()[0], inplace=True)
            st.success("Missing values filled!")
            st.session_state["df"] = df

        # Detect Outliers
        st.subheader("4️⃣ Detecting Outliers using Z-Score")
        z_scores = np.abs(stats.zscore(df.select_dtypes(include=["number"])))
        outliers = (z_scores > 3).sum()
        st.write(outliers)

        if st.button("Remove Outliers"):
            df = df[(z_scores < 3).all(axis=1)]
            st.success("Outliers removed!")
            st.session_state["df"] = df

        # Encode Categorical Variables
        st.subheader("5️⃣ Encoding Categorical Variables")
        categorical_cols = df.select_dtypes(include=["object"]).columns
        if len(categorical_cols) > 0:
            encoding_option = st.radio("Choose encoding type:", ["Label Encoding", "One-Hot Encoding"])
            if st.button("Apply Encoding"):
                if encoding_option == "Label Encoding":
                    le = LabelEncoder()
                    for col in categorical_cols:
                        df[col] = le.fit_transform(df[col])
                else:
                    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
                st.success("Categorical encoding applied!")
                st.session_state["df"] = df

        # Feature Scaling
        st.subheader("6️⃣ Feature Scaling (Normalization)")
        if st.button("Apply Min-Max Scaling"):
            scaler = MinMaxScaler()
            df[df.select_dtypes(include=["number"]).columns] = scaler.fit_transform(df.select_dtypes(include=["number"]))
            st.success("Min-Max Scaling Applied!")
            st.session_state["df"] = df

        # Remove Duplicates
        st.subheader("7️⃣ Removing Duplicate Entries")
        duplicate_count = df.duplicated().sum()
        st.write(f"Duplicate Rows Found: {duplicate_count}")

        if st.button("Remove Duplicates"):
            df = df.drop_duplicates()
            st.success("Duplicates Removed!")
            st.session_state["df"] = df

        # Feature Engineering
        st.subheader("8️⃣ Feature Engineering")
        if "date" in df.columns:
            df["Year"] = pd.to_datetime(df["date"]).dt.year
            df["Month"] = pd.to_datetime(df["date"]).dt.month
            st.success("Extracted Year and Month from Date!")

        # Train-Test Split
        st.subheader("9️⃣ Train-Test Split")
        target_col = st.selectbox("Select Target Column", df.columns)

        if st.button("Split Data"):
            X = df.drop(target_col, axis=1)
            y = df[target_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            st.session_state["X_train"], st.session_state["X_test"] = X_train, X_test
            st.session_state["y_train"], st.session_state["y_test"] = y_train, y_test
            st.success("Data Split into Training and Testing Sets!")

        # PCA Dimensionality Reduction
        st.subheader("🔟 Dimensionality Reduction using PCA")

        if "X_train" in st.session_state:
            if st.button("Apply PCA (2 Components)"):
                pca = PCA(n_components=2)
                X_train = st.session_state["X_train"]  # Retrieve X_train
                X_pca = pca.fit_transform(X_train)

                df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
                st.session_state["df_pca"] = df_pca
                st.success("PCA Applied!")

                # Scatter Plot
                fig, ax = plt.subplots()
                sns.scatterplot(x=df_pca["PC1"], y=df_pca["PC2"])
                st.pyplot(fig)
        else:
            st.error("Please perform the Train-Test Split before applying PCA.")

        # Save Processed Data
        st.subheader("📥 Download Preprocessed Data")
        processed_csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(label="Download Processed Data", data=processed_csv, file_name="processed_data.csv", mime="text/csv")

        if st.button("Next"):
            go_to_page("train_model")

# --- PAGE 4: MODEL TRAINING (Future Scope) ---
elif st.session_state["page"] == "train_model":
    st.title("Model Training 🔥")
    st.write("🚧 This feature is under development...")