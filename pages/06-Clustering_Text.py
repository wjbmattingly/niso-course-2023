import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import umap.umap_ as umap
import plotly.express as px
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Load the sentence-transformers model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to load data
@st.cache_data
def load_data():
    df = pd.read_parquet('data/processed_texts.parquet')
    return df

# Function to preprocess documents based on user selection
def preprocess_documents(df, preprocess_type):
    if preprocess_type == 'lemma':
        return df['text_no_punct_lemma'].tolist()
    elif preprocess_type == 'lemma_and_punct':
        return df['text_no_punct'].tolist()
    elif preprocess_type == 'punct':
        return df['original_text'].tolist()
    elif preprocess_type == 'word_embeddings':
        return df['ml_vectors'].tolist()

st.title('Clustering of 20 Newsgroups Data with UMAP and Various Preprocessing')

# Sidebar options
st.sidebar.header('Options')
selected_preprocess = st.sidebar.selectbox('Select preprocessing method', 
                                           ('lemma', 'lemma_and_punct', 'punct', 'word_embeddings'))
num_clusters = st.sidebar.slider('Select number of clusters', min_value=2, max_value=20, value=5)

# Load data
df = load_data()
st.write(f'Loaded {len(df)} documents.')

if st.sidebar.button("Cluster Data"):
    with st.spinner('Processing data...'):
        documents = preprocess_documents(df, selected_preprocess)

        # Vectorization is not needed for word embeddings
        if selected_preprocess != 'word_embeddings':
            vectorizer = TfidfVectorizer(stop_words='english')
            X = vectorizer.fit_transform(documents)
        else:
            X = np.array(documents)

        # Clustering
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(X)
        labels = kmeans.labels_

        # Dimensionality reduction
        reducer = umap.UMAP()
        X_reduced = reducer.fit_transform(X)

        df['UMAP1'] = X_reduced[:, 0]
        df['UMAP2'] = X_reduced[:, 1]
        df['cluster'] = labels
        if selected_preprocess != 'word_embeddings':
            df['text'] = documents
        else:
            df['text'] = df['original_text']

        # Plot
        fig = px.scatter(df, x='UMAP1', y='UMAP2', color='cluster', hover_data=['text'])
        fig.update_traces(marker=dict(size=7, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))
        fig.update_layout(legend=dict(title='Cluster'), hovermode='closest')
        st.plotly_chart(fig)
        st.markdown('**Note:** Hover over points in the plot to see the text.')
