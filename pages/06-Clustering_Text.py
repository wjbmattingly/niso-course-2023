import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import umap.umap_ as umap
from sklearn.datasets import fetch_20newsgroups
import plotly.express as px
import pandas as pd
import string
import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to load data
@st.cache_data
def load_data(categories):
    newsgroups_data = fetch_20newsgroups(subset='all', categories=categories,
                                         remove=('headers', 'footers', 'quotes'))
    return newsgroups_data.data[:200]

# Function to preprocess documents
def preprocess_documents(documents, lemmatize, remove_punctuation):
    preprocessed_documents = []
    for doc in nlp.pipe(documents, disable=["ner", "parser"]): # Disable unnecessary pipelines
        if lemmatize:
            if remove_punctuation:
                preprocessed_documents.append(" ".join(token.lemma_ for token in doc if token.lemma_ != '-PRON-' and token.is_punct != True))
            else:
                preprocessed_documents.append(" ".join(token.lemma_ for token in doc if token.lemma_ != '-PRON-'))
        else:
            if remove_punctuation:
                preprocessed_documents.append(" ".join(token.text for token in doc if token.is_punct != True))
            else:
                preprocessed_documents.append(" ".join(token.text for token in doc))
    return preprocessed_documents

st.title('TF-IDF Clustering of 20 Newsgroups Data with UMAP')

# Sidebar options
st.sidebar.header('Options')
selected_categories = ["rec.sport.baseball", "sci.space"]
num_clusters = st.sidebar.slider('Select number of clusters', min_value=2, max_value=20, value=5)
lemmatize = st.sidebar.checkbox('Lemmatize text', value=True)
remove_punctuation = st.sidebar.checkbox('Remove punctuation', value=True)

# Load and preprocess data
documents = load_data(selected_categories)
st.write(f'Loaded {len(documents)} documents.')

if st.sidebar.button("Preprocess and Cluster Data"):
    with st.spinner('Preprocessing data...'):
        documents = preprocess_documents(documents, lemmatize, remove_punctuation)
        st.success('Preprocessing complete.')

    # Proceed with vectorization and clustering
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(documents)

    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(X)
    labels = kmeans.labels_

    reducer = umap.UMAP()
    X_reduced = reducer.fit_transform(X)

    df = pd.DataFrame({'UMAP1': X_reduced[:, 0], 'UMAP2': X_reduced[:, 1], 'cluster': labels, 'text': documents})

    # Plot
    fig = px.scatter(df, x='UMAP1', y='UMAP2', color='cluster', hover_data=['text'])
    fig.update_traces(marker=dict(size=7, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))
    fig.update_layout(legend=dict(title='Cluster'), hovermode='closest')
    st.plotly_chart(fig)
    st.markdown('**Note:** Hover over points in the plot to see the text.')
