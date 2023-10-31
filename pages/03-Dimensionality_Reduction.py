import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from umap import UMAP
import seaborn as sns

# Load dataset
@st.cache_resource
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

@st.cache_resource
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

@st.cache_resource
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
all_features = df.columns.tolist()
selected_features = st.sidebar.multiselect(
    'Select features to reduce', 
    options=all_features,
    default=all_features
)

# Sidebar - Dimensionality Reduction Choice
st.sidebar.header('Dimensionality Reduction')
reduction_method = st.sidebar.radio("Choose a dimensionality reduction method:", ('PCA', 't-SNE', 'UMAP'))

# Sidebar - Color selection
color_column = st.sidebar.selectbox('Select column for color coding:', options=df.columns)

# Main - Show dataframe
main_expander = st.expander("Dataset")
main_expander.write(df)

# Preprocessing function
def preprocess_df(df, features):
    df_processed = df[features].copy()
    
    # Encode categorical features if selected
    for feature in features:
        if df_processed[feature].dtype == 'object' or df_processed[feature].dtype.name == 'category':
            le = LabelEncoder()
            df_processed[feature] = le.fit_transform(df_processed[feature].astype(str))
    
    # Scale features
    scaler = StandardScaler()
    df_processed = pd.DataFrame(scaler.fit_transform(df_processed), columns=features)
    
    return df_processed

# Perform dimensionality reduction
def reduce_dimensions(method, df, features, additional_hover_data=None):

    processed_df = preprocess_df(df, features)

    if method == 'PCA':
        model = PCA(n_components=2)
    elif method == 't-SNE':
        model = TSNE(n_components=2, random_state=42)
    elif method == 'UMAP':
        model = UMAP(n_components=2, random_state=42)
    
    result = model.fit_transform(processed_df)
    reduced_data = pd.DataFrame(result, columns=['Dim1', 'Dim2'])

    # Merge the reduced data with the original selected features for hover information
    hover_data = df[features].reset_index(drop=True)
    if additional_hover_data:
        hover_data = pd.concat([hover_data, df[additional_hover_data].reset_index(drop=True)], axis=1)
    reduced_data = pd.concat([reduced_data, hover_data], axis=1)
    
    return reduced_data

# Now when calling the plotting function, you should have all the selected features available for hover
# if color_column not in selected_features:
#     additional_hover_columns = [color_column]
# else:
#     additional_hover_columns = []

additional_hover_columns = [column for column in df.columns if column not in selected_features]
# Perform and plot dimensionality reduction
if selected_features:
    st.write(f"## Dimensionality Reduction using {reduction_method}")
    reduced_data = reduce_dimensions(reduction_method, df, selected_features, additional_hover_columns)

    # Now, ensure all columns for hover data are included, regardless of selection
    all_hover_data = selected_features + additional_hover_columns

    # Use Plotly for interactive scatter plot
    fig = px.scatter(
        reduced_data,
        x='Dim1',
        y='Dim2',
        color=reduced_data[color_column].astype(str) if color_column in reduced_data.columns else None,
        hover_data=all_hover_data
    )
    fig.update_traces(marker=dict(size=10),
                      selector=dict(mode='markers'))
    fig.update_layout(title=f'{reduction_method} Results', title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)

