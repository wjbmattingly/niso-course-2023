import streamlit as st
import pandas as pd
import seaborn as sns
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px

# Load dataset
@st.cache_data
def load_titanic_data():
    data = sns.load_dataset('titanic')

    
    # You can fill numeric NaN with mean or use another imputation strategy
    num_cols = data.select_dtypes(include=['number']).columns
    data[num_cols] = data[num_cols].fillna(data[num_cols].mean())
    
    # For categorical data, fill with mode (most frequent)
    cat_cols = data.select_dtypes(include=['object', 'category']).columns
    data[cat_cols] = data[cat_cols].fillna(data[cat_cols].mode().iloc[0])

    # Transform the age column
    age_bins = [0, 12, 19, 29, 39, 49, 59, 69, 79, 89, 99]
    age_labels = ['child', 'teen', 'twenties', 'thirties', 'forties', 'fifties', 'sixties', 'seventies', 'eighties', 'nineties']
    data['age_group'] = pd.cut(data['age'], bins=age_bins, labels=age_labels, right=False)
    data['age_group'] = data['age_group'].astype(str)  # Convert to string

    # Group the fare column into bins of every 5
    fare_max = data['fare'].max()
    fare_bins = list(range(0, int(fare_max) + 20, 20))
    data['fare_group'] = pd.cut(data['fare'], bins=fare_bins)
    data['fare_group'] = data['fare_group'].astype(str)  # Convert to string
    exclude_columns = ['class', 'who', 'alive', 'age', 'fare', 'embarked']
    data = data.drop(columns=exclude_columns)

    data = data.reset_index(drop=True)
    return data

@st.cache_data
def load_iris_data():
    # Load the iris dataset
    data = sns.load_dataset('iris')
    
    # Fill numeric NaNs with the mean (if any, though the Iris dataset typically has none)
    num_cols = data.select_dtypes(include=['number']).columns
    data[num_cols] = data[num_cols].fillna(data[num_cols].mean())
    
    # Bin numerical data into groups and create new columns for these groups
    # Define bins for sepal_length
    sepal_length_bins = [4, 5, 6, 7, 8]
    sepal_length_labels = ['4-5', '5-6', '6-7', '7-8']
    data['sepal_length_group'] = pd.cut(data['sepal_length'], bins=sepal_length_bins, labels=sepal_length_labels, include_lowest=True)
    
    # Define bins for petal_length
    petal_length_bins = [1, 2, 3, 4, 5, 6, 7]
    petal_length_labels = ['1-2', '2-3', '3-4', '4-5', '5-6', '6-7']
    data['petal_length_group'] = pd.cut(data['petal_length'], bins=petal_length_bins, labels=petal_length_labels, include_lowest=True)
    
    # Convert binned data to string if necessary
    data['sepal_length_group'] = data['sepal_length_group'].astype(str)
    data['petal_length_group'] = data['petal_length_group'].astype(str)
    
    # # Drop the original numeric columns
    # data.drop(columns=['sepal_length', 'petal_length'], inplace=True)
    
    data = data.reset_index(drop=True)
    return data

@st.cache_data
def load_planets_data():
    # Load the dataset from Seaborn
    data = sns.load_dataset('planets')

    # Handle missing values, for example, by filling with the mean or median
    num_cols = data.select_dtypes(include=['number']).columns
    data[num_cols] = data[num_cols].fillna(data[num_cols].mean())

    # Bin numerical columns as needed
    # For example, binning the 'orbital_period' into categories
    orbital_period_bins = [0, 365, 365*2, 365*10, 365*100, max(data['orbital_period'].max(), 365*1000)]
    orbital_period_labels = ['< 1 year', '1-2 years', '2-10 years', '10-100 years', '> 100 years']
    data['orbital_period_group'] = pd.cut(data['orbital_period'], bins=orbital_period_bins, labels=orbital_period_labels, right=False)
    data['orbital_period_group'] = data['orbital_period_group'].astype(str)  # Convert to string for easier visualization

    # Bin 'distance' into categories, e.g., every 50 light-years
    distance_bins = list(range(0, int(data['distance'].max()) + 100, 50))
    data['distance_group'] = pd.cut(data['distance'], bins=distance_bins)
    data['distance_group'] = data['distance_group'].astype(str)  # Convert to string

    # Drop the original numerical columns if they are not needed anymore
    exclude_columns = ['orbital_period', 'distance']
    data = data.drop(columns=exclude_columns)

    # Reset index if needed
    data = data.reset_index(drop=True)
    
    return data


# Encoding categorical features
def encode_features(df, features):
    le = LabelEncoder()
    for feature in features:
        df[feature] = le.fit_transform(df[feature])
    return df

# Create a dictionary mapping dataset names to their loaders
dataset_loaders = {
    "Titanic": load_titanic_data,
    "Iris": load_iris_data,
    "Planets": load_planets_data
}

# Use the dictionary to select and load the dataset
dataset = st.sidebar.selectbox("Select Dataset", list(dataset_loaders.keys()))
df = dataset_loaders[dataset]()

# Sidebar - Feature selection
st.sidebar.header('Feature Selection')
numeric_features = df.columns.tolist()
numeric_features = [n for n in numeric_features if "UMAP" not in n and "cluster" not in n]
selected_features = st.sidebar.multiselect('Select features for dimensionality reduction:', options=numeric_features, default=numeric_features)

# Sidebar - UMAP parameters
st.sidebar.header('UMAP Parameters')
n_neighbors = st.sidebar.slider('n_neighbors', 2, 200, 15)
min_dist = st.sidebar.slider('min_dist', 0.0, 1.0, 0.1)
# n_components = st.sidebar.slider('n_components', 2, 5, 2)
n_components = 2
metric = st.sidebar.selectbox('metric', ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'canberra', 'braycurtis', 'mahalanobis'])

# Sidebar - HDBSCAN parameters
st.sidebar.header('HDBSCAN Parameters')
min_cluster_size = st.sidebar.slider('min_cluster_size', 5, 50, 15)
min_samples = st.sidebar.slider('min_samples', 1, 50, 5)

# Perform UMAP reduction
# Perform UMAP reduction
# st.sidebar.subheader('Run UMAP and HDBScan')
if st.sidebar.button('Run UMAP and HDBScan'):
    # Encode categorical features before scaling
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_features:
        df = encode_features(df, categorical_features)
        
    # Ensure only selected numeric features are scaled
    numeric_features_selected = [feature for feature in selected_features if feature in df.select_dtypes(include=['number']).columns]
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numeric_features_selected])
    
    umap_reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric)
    embedding = umap_reducer.fit_transform(scaled_data)
    
    # Add the UMAP components to the dataframe
    for i in range(n_components):
        df[f'UMAP_{i+1}'] = embedding[:, i]

    hdbscan_cluster = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_features:
        df = encode_features(df, categorical_features)
    df['cluster'] = hdbscan_cluster.fit_predict(df[selected_features])

    # Main - Show dataframe
    main_expander = st.expander("Dataset")
    main_expander.write(df)

    # Visualization

    fig = px.scatter(df, x='UMAP_1', y='UMAP_2', color='cluster', hover_data=df.columns)
    st.plotly_chart(fig)