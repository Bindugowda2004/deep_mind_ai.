import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 📌 Function to extract keywords from dataset
def extract_keywords(df):
    keywords = df.columns.tolist()  # Extract column names as keywords
    common_terms = ["id", "date", "time", "value", "name", "amount", "index"]  # Remove generic terms
    filtered_keywords = [word for word in keywords if word.lower() not in common_terms]
    return filtered_keywords[:5]  # Return top 5 keywords

# 📌 Function to fetch related research papers
def fetch_research_papers(keywords):
    query = " ".join(keywords) + " research paper"
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=5&fields=title,url,abstract"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("data", [])
    except requests.RequestException:
        return []

# 📌 Function to compute similarity between dataset and research papers
def compute_similarity(dataset_text, papers):
    if not papers:
        return []

    paper_texts = [str(paper.get("abstract") or "").strip() for paper in papers]
    dataset_text = str(dataset_text).strip()

    # Keep only non-empty abstracts
    valid_papers = [(paper, text) for paper, text in zip(papers, paper_texts) if text]
    valid_paper_texts = [text for _, text in valid_papers]

    if not valid_paper_texts:  
        return [0] * len(papers)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([dataset_text] + valid_paper_texts)
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    return similarity_scores, valid_papers  # Return valid papers along with scores

# 📌 Function to fetch latest project recommendations
def fetch_project_recommendations(keywords):
    query = " ".join(keywords) + " latest project ideas"
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=5&fields=title,url"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("data", [])
    except requests.RequestException:
        return []

# 📌 Streamlit UI
st.title("📄 AI-Powered Research Paper & Project Finder")
st.write("Upload your dataset, and we'll find related research papers and latest project recommendations based on its content.")

# 📂 File Upload
uploaded_file = st.file_uploader("📂 Upload Your Dataset (CSV)", type=["csv"])

if uploaded_file:
    st.success("✅ File Uploaded Successfully!")

    # Read dataset
    df = pd.read_csv(uploaded_file)
    dataset_text = " ".join(df.astype(str).values.flatten())  # Convert dataset content into text
    
    # 🔍 Extract Keywords
    keywords = extract_keywords(df)
    st.subheader("🔍 Extracted Keywords from Dataset:")
    st.write(", ".join(keywords))

    # 📚 Fetch Research Papers
    st.subheader("📚 Related Research Papers")
    papers = fetch_research_papers(keywords)

    if papers:
        similarity_scores, valid_papers = compute_similarity(dataset_text, papers)
        
        for (paper, score) in zip(valid_papers, similarity_scores):
            similarity_percentage = score * 100
            paper_title = paper[0].get("title", "No Title Available")
            paper_url = paper[0].get("url", "#")
            paper_abstract = str(paper[0].get("abstract") or "No abstract available.")

            st.markdown(f"### [{paper_title}]({paper_url})")
            st.write(f"📝 {paper_abstract[:250]}...")  # Show first 250 chars
            st.write(f"📊 Similarity Score: *{similarity_percentage:.2f}%*")
            st.write("---")
    else:
        st.warning("⚠ No related research papers found. Try different dataset keywords.")
    
    # 🚀 Fetch Latest Project Recommendations
    st.subheader("🚀 Latest Project Recommendations")
    projects = fetch_project_recommendations(keywords)

    if projects:
        for project in projects:
            project_title = project.get("title", "No Title Available")
            project_url = project.get("url", "#")

            st.markdown(f"### [{project_title}]({project_url})")
            st.write("---")
    else:
        st.warning("⚠ No latest project recommendations found. Try different dataset keywords.")