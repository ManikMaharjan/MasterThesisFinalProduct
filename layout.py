import streamlit as st
import pandas as pd
import pickle, json
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.graph_objects as go


@st.cache_data
def load_model():
    with open('dumps/kprototype.dat', 'rb') as f:
        kproto = pickle.load(f)


    with open('dumps/kmeans.dat', 'rb') as f:
        kmeans = pickle.load(f)

    with open('dumps/encodings.dat', 'rb') as f:
        encodings = pickle.load(f)

    return encodings, kproto, kmeans


@st.cache_data
def load_data():
    return pd.read_csv('okcupid_profiles_preprocessed.csv')

def column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols,query_cols,sorter=sidx)]

def plotUmap(embeddings, clusters):
    k_proto_umap = pd.DataFrame({'x':embeddings[0][:,0],
                             'y':embeddings[0][:,1],
                             'clusters': clusters,
                            })

    k_proto_umap['clusters'] = k_proto_umap['clusters'].apply(str)

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.scatterplot(data=k_proto_umap, x='x', y='y', hue='clusters', palette='tab10', ax=ax, alpha=0.8, s=50)
    ax.set_title(f'OkCupid dataset clusters with {len(k_proto_umap["clusters"].unique())} clusters')

    st.pyplot(fig)

encodings, kproto, kmeans = load_model()
df = load_data()

features = ['age',
 'height',
 'status',
 'sex',
 'orientation',
 'body_type_',
 'education_',
 'drinks_',
 'drugs_',
 'smokes_',
 'likes_dogs_',
 'likes_cats_']

dfFeatures = df[features]

cat_cols = dfFeatures.select_dtypes(include='object')
categorical_indices = list(column_index(dfFeatures, cat_cols.columns))


def encodeDf(df):

    for col in df.columns:
        if df[col].dtype == 'object':
            lc = encodings[col]

            df[col] = lc.transform(df[col])
        
        scaler = StandardScaler()
        df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))


def plot_pca(df, clusters, n_components=2):
    df = df.copy()
    
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(df)
    principalDf = pd.DataFrame(data = principalComponents, 
                    columns = ['PC1', 'PC2']
        )
    
    principalDf['cluster'] = clusters.astype(str)
    # don't have continuous color scale

    last_row = principalDf.iloc[-1]
    rest_df = principalDf.iloc[:-1]

    # Create a figure
    fig = go.Figure()

    # Plot all existing data points
    fig.add_trace(go.Scatter(
        x=rest_df['PC1'], 
        y=rest_df['PC2'],
        mode='markers',
        marker=dict(size=7, color=rest_df['cluster'].astype(int), colorscale='Viridis', opacity=0.6),
        text=rest_df['cluster'],
        name="Existing Data"
    ))

    # Plot the last row separately in red
    fig.add_trace(go.Scatter(
        x=[last_row['PC1']], 
        y=[last_row['PC2']],
        mode='markers',
        marker=dict(size=10, color='red', symbol='star'),
        text=["Your data point"],
        name="User's data"
    ))

    
    # last 
    st.plotly_chart(fig)
    

def plotClusters(df, model, row):
    df = df.copy()
    df.loc[len(df)] = row

    encodeDf(df)

    clusters = model.predict(df)

    st.write(f"Predicted Cluster using Kmeans: {clusters[-1]}")
    plot_pca(df, clusters)



def submitParameters():
    st.write("## **Clustering user via Kprototype and Kmean.....**")

    # Calculate the growth

    age = st.session_state.age
    height = st.session_state.height
    status = st.session_state.status
    sex = st.session_state.sex
    orientation = st.session_state.orientation
    body_type = st.session_state.body_type
    education = st.session_state.education
    drinks = st.session_state.drinks
    drugs = st.session_state.drugs
    smokes = st.session_state.smokes
    likes_dogs = "Yes" if st.session_state.likes_dogs else "No"
    likes_cats = "Yes" if st.session_state.likes_cats else "No"

    array = [age, height, status, sex, orientation, body_type, education, drinks, drugs, smokes, likes_dogs, likes_cats]

    st.write(f"User Profile: {array}")

    pred = kproto.predict(np.asarray([array]), categorical=categorical_indices)
    
    st.write(f"Predicted Cluster using Kprototype: {pred}")

    plotClusters(dfFeatures, kmeans, array)

    # pred = model.predict([[consumers, transactions, transaction_size, network_traffic]])[0]
    # st.write(f'Predicted growth percentage:     **{round(pred * 100, 4)} %**')
    # st.write(f'Database expected to reach       **{round(rough_db_size + rough_db_size * pred, 4)}** GB')

    # newDf = df.copy()
    # newDf.loc[len(newDf)] = [consumers, transactions, transaction_size, network_traffic, pred]

    # fig = px.scatter(newDf, x='users', y='traffic', size='transactions', color='growth-rate',
    #                  hover_name='transactions', size_max=10,
    #                  labels={
    #                      "users": "Number of Users",
    #                      "traffic": "Website Traffic",
    #                      "transactions": "Number of Transactions",
    #                      "growth-rate": "Growth Rate"
    #                  },
    #                  title="User Base vs. Traffic with Transactions and Growth Rate")
    
    # # Add unique marker for the new data point
    # fig.add_scatter(x=[consumers], y=[network_traffic], mode='markers',
    #                 marker=dict(size=15, color='red', symbol='x'),
    #                 name='Predicted Data Point')

    # st.plotly_chart(fig)




    
st.write("## **OkCupid Clustering Demo**")
st.write("This app performs clustering on OkCupid user data based on various profile attributes, such as age, height, relationship status, and personal habits. Use the sidebar to input values and explore clustering results.")

# Sidebar for input fields
st.sidebar.header('User Profile Inputs')

age = st.sidebar.slider('Age', min_value=18, max_value=99, value=25, step=1, key='age')
height = st.sidebar.slider('Height (cm)', min_value=140, max_value=220, value=170, step=1, key='height')
status = st.sidebar.selectbox('Relationship Status', encodings["status"].classes_, key='status')
sex = st.sidebar.selectbox('Gender', encodings["sex"].classes_, key='sex')

orientation = st.sidebar.selectbox('Sexual Orientation', encodings["orientation"].classes_, key='orientation')
body_type = st.sidebar.selectbox('Body Type', encodings["body_type_"].classes_, key='body_type')

education = st.sidebar.selectbox('Education Level', encodings["education_"].classes_, key='education')
drinks = st.sidebar.selectbox('Drinks Alcohol', encodings["drinks_"].classes_, key='drinks')
drugs = st.sidebar.selectbox('Uses Drugs', encodings["drugs_"].classes_, key='drugs')
smokes = st.sidebar.selectbox('Smokes', encodings["smokes_"].classes_, key='smokes')
likes_dogs = st.sidebar.checkbox('Likes Dogs', key='likes_dogs')
likes_cats = st.sidebar.checkbox('Likes Cats', key='likes_cats')



submit = st.sidebar.button("Cluster User", on_click=submitParameters)
