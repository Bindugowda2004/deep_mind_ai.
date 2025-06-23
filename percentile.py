import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Function to search for research papers using Semantic Scholar API
def fetch_research_papers(query):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&fields=title,abstract,url"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("data", [])
    return []

# Function to compute Jaccard Similarity
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

# Function to compute Cosine Similarity on numerical data
def compute_cosine_similarity(df1, df2):
    scaler = StandardScaler()
    df1_scaled = scaler.fit_transform(df1)
    df2_scaled = scaler.transform(df2)
    return cosine_similarity(df1_scaled, df2_scaled)[0][0]

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files['file']
    filename = file.filename
    filepath = os.path.join("uploads", filename)
    file.save(filepath)
    
    # Load dataset
    user_dataset = pd.read_csv(filepath)
    user_attributes = set(user_dataset.columns)
    
    # Fetch relevant research papers based on attributes
    query = ", ".join(user_attributes)
    research_papers = fetch_research_papers(query)
    
    results = []
    
    # Compare with research papers
    for paper in research_papers:
        if "abstract" not in paper:
            continue
        paper_attributes = set(paper['abstract'].split()[:20])  # Extract first 20 words as attributes
        jaccard_sim = jaccard_similarity(user_attributes, paper_attributes)
        
        # If numerical data, compute cosine similarity
        cosine_sim = 0
        if not user_dataset.select_dtypes(include=[np.number]).empty:
            paper_df = pd.DataFrame(np.random.rand(len(user_dataset), len(paper_attributes)), columns=list(paper_attributes))
            cosine_sim = compute_cosine_similarity(user_dataset.select_dtypes(include=[np.number]), paper_df)
        
        similarity_score = (jaccard_sim + cosine_sim) / 2  # Weighted similarity
        
        results.append({
            "title": paper['title'],
            "url": paper['url'],
            "similarity_percentage": round(similarity_score * 100, 2)
        })
    
    return jsonify({"results": results})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)