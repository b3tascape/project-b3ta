### PROJECT-B3TA-STREAMLIT

#######################################################################################################################################
#######################################################################################################################################

### import libraries
import pandas as pd
import numpy as np
## Pandas has a depedency on NumPy so loads automatically but best practice to load full NumPy package
print(f"Numpy version: {np.__version__}")
print(f"pandas version: {pd.__version__}")

# streamlit
import streamlit as st
import joblib

# visualisation
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default = 'notebook'

# sklearn
import sklearn as sk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, f_regression
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
print(f"skLearn version: {sk.__version__}")

# scipy
import scipy
from scipy.stats import norm

# statsmodels
import statsmodels.api as sm

# other
import numbers
import requests
import json
from xgboost import XGBClassifier

# b3ta functions
import b3tafunc as b3


#######################################################################################################################################
#######################################################################################################################################

### IMPORTS

# full df - raw
#df = pd.read_csv('../data/05-b3-exports/df.csv', index_col=0)
#df = pd.read_csv('../data/03-b3-exports/03-b3_df.csv', index_col=0)
df50 = pd.read_csv('../data/06-b3-streamlit-git/06-b3_df50-sl.csv', index_col=0)
user_raw_df = pd.read_csv('../data/06-b3-streamlit-git/04-b3-user_raw_df.csv', index_col=0)

# create sound profile dfs
london_snd = df50[df50['sound_profile'] == 'london']
manchester_snd = df50[df50['sound_profile'] == 'manchester']
ibiza_snd = df50[df50['sound_profile'] == 'ibiza']
berlin_snd = df50[df50['sound_profile'] == 'berlin']
kingston_snd = df50[df50['sound_profile'] == 'kingston']
nyc_snd = df50[df50['sound_profile'] == 'nyc']
la_snd = df50[df50['sound_profile'] == 'la']
combined_snd = df50[df50['sound_profile'] != 'unknown']

# upsampled_rf
london_upsamp_test_rf = pd.read_csv('../data/06-b3-streamlit-git/london_upsamp_test_rf-mergedv1.csv', index_col=0)
manchester_upsamp_test_rf = pd.read_csv('../data/06-b3-streamlit-git/manchester_upsamp_test_rf-mergedv1.csv', index_col=0)
ibiza_upsamp_test_rf = pd.read_csv('../data/06-b3-streamlit-git/ibiza_upsamp_test_rf-mergedv1.csv', index_col=0)
berlin_upsamp_test_rf = pd.read_csv('../data/06-b3-streamlit-git/berlin_upsamp_test_rf-mergedv1.csv', index_col=0)
kingston_upsamp_test_rf = pd.read_csv('../data/06-b3-streamlit-git/kingston_upsamp_test_rf-mergedv1.csv', index_col=0)
nyc_upsamp_test_rf = pd.read_csv('../data/06-b3-streamlit-git/nyc_upsamp_test_rf-mergedv1.csv', index_col=0)
la_upsamp_test_rf = pd.read_csv('../data/06-b3-streamlit-git/la_upsamp_test_rf-mergedv1.csv', index_col=0)

# upsampled_cs
london_upsamp_test_cs = pd.read_csv('../data/06-b3-streamlit-git/london_upsamp_test_cs-mergedv1.csv', index_col=0)
manchester_upsamp_test_cs = pd.read_csv('../data/06-b3-streamlit-git/manchester_upsamp_test_cs-mergedv1.csv', index_col=0)
ibiza_upsamp_test_cs = pd.read_csv('../data/06-b3-streamlit-git/ibiza_upsamp_test_cs-mergedv1.csv', index_col=0)
berlin_upsamp_test_cs = pd.read_csv('../data/06-b3-streamlit-git/berlin_upsamp_test_cs-mergedv1.csv', index_col=0)
kingston_upsamp_test_cs = pd.read_csv('../data/06-b3-streamlit-git/kingston_upsamp_test_cs-mergedv1.csv', index_col=0)
nyc_upsamp_test_cs = pd.read_csv('../data/06-b3-streamlit-git/nyc_upsamp_test_cs-mergedv1.csv', index_col=0)
la_upsamp_test_cs = pd.read_csv('../data/06-b3-streamlit-git/la_upsamp_test_cs-mergedv1.csv', index_col=0)


#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################


### SIDEBAR - TITLE ### ### ### ### ### ### ### ### ###

# Create a sidebar section for user input

st.sidebar.subheader("_g3ocorrect.")

st.sidebar.write('==') 


### SIDEBAR - GEOCORRECTOR - SELECT LOCATION ### ### ### ### ### ### ### ### ###

sp_location = st.sidebar.radio(
    "location:",
    ["london", "manchester", "nyc", "la", "kingston", "ibiza", "berlin", "global"])

st.sidebar.write('==') 



### SIDEBAR - GEOCORRECTOR - SELECT ALGO ### ### ### ### ### ### ### ### ### ###

sp_algo = st.sidebar.radio(
    "algo:",
    ['1', '2', '3'])

st.sidebar.write('==') 



### SIDEBAR - GEOCORRECTOR - SELECT TRACKS ### ### ### ### ### ### ### ### ### ### ### ###

# Create an empty dataframe to store the selected tracks
selected_tracks_df = pd.DataFrame(columns=df50.columns)

# Define the number of tracks to select
num_tracks_to_select = 5

# Loop to select tracks
for i in range(num_tracks_to_select):
    # Create selectbox widgets with autocomplete functionality for artist names
    selected_artist_names = st.sidebar.multiselect(f"artist {i + 1}", df50['artist_name'].unique())

    # Filter the main dataframe based on the selected artist names
    artist_filtered_df = df50[df50['artist_name'].isin(selected_artist_names)]

    # Create selectbox widgets with autocomplete functionality for track names
    selected_track_names = st.sidebar.multiselect(f"track {i + 1}", artist_filtered_df['track_name'].unique())

    # Filter further based on the selected track names
    filtered_tracks_df = artist_filtered_df[artist_filtered_df['track_name'].isin(selected_track_names)]
    
    # Add only the first result if there are multiple matches
    if len(filtered_tracks_df) > 1:
        filtered_tracks_df = filtered_tracks_df.head(1)

    # Add the selected track to the new dataframe
    selected_tracks_df = pd.concat([selected_tracks_df, filtered_tracks_df])

    # Convert duration_ms column to integer data type
    selected_tracks_df['duration_ms'] = selected_tracks_df['duration_ms'].astype('int')

# Add a separator in the sidebar
st.sidebar.write('---')



### SIDEBAR - INSIGHTS - SELECT BOXPLOT METRIC ### ### ### ### ### ### ### ### ### ### ### ###
    
st.sidebar.subheader("_insights.")

metric = st.sidebar.radio(
    "boxplot metric:",
    ["duration_ms", "tempo", "loudness", "energy", "valence", "danceability", "speechiness", "instrumentalness", "acousticness", "liveness"])


### SIDEBAR - INSIGHTS - GENRE SUNBURST ### ### ### ### ### ### ### ### ### ### ### ###

#genre_location = st.sidebar.radio(
#    "sunburst location:",
#    ["london", "manchester", "nyc", "kingston", "ibiza", "berlin", "la", "global"])

st.sidebar.write('---')



### SIDEBAR - G3TSIMILAR ### ### ### ### ### ### ### ### ### ### ### ### ### ###

#st.sidebar.subheader("_g3tsimilar.")

# Create selectbox widgets with autocomplete functionality for artist names
#selected_artist_names = st.sidebar.multiselect(f"artist", df50['artist_name'].unique())

# Filter the main dataframe based on the selected artist names
#artist_filtered_df = df50[df50['artist_name'].isin(selected_artist_names)]

# Create selectbox widgets with autocomplete functionality for track names
#selected_track_names = st.sidebar.multiselect(f"track", artist_filtered_df['track_name'].unique())

# Filter further based on the selected track names

#filtered_df = artist_filtered_df[artist_filtered_df['track_name'].isin(selected_track_names)]
# Add the selected track to the new dataframe
#selected_tracks_df = pd.concat([selected_tracks_df, filtered_df])

# Convert a single column to a specific data type
#selected_tracks_df['duration_ms'] = selected_tracks_df['duration_ms'].astype('int')

#st.sidebar.write('---')



###############################################################################################
###############################################################################################


### MAIN - TITLE ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

st.write('#### /// project-b3ta ///')  

st.write('music recommendations that change with the world around us.')  

st.write('---')   




### MAIN - GEOCORRECTOR - TRACK LISTS ### ### ### ### ### ### ### ### ### ### ### ###

# User playlist header
st.write('_user.') 

# Convert duration_ms to minutes and seconds - user
selected_tracks_df['duration_minutes'] = selected_tracks_df['duration_ms'] // 60000  # Get minutes
selected_tracks_df['duration_seconds'] = (selected_tracks_df['duration_ms'] // 1000) % 60  # Get remaining seconds
selected_tracks_df['duration'] = selected_tracks_df['duration_minutes'].astype(str) + ':' + selected_tracks_df['duration_seconds'].astype(str).str.zfill(2)  # Combine minutes and seconds into a single column
selected_tracks_df.drop(['duration_minutes', 'duration_seconds'], axis=1, inplace=True)  # Drop temp columns

# Round Tempo - user
selected_tracks_df['tempo'] = round(selected_tracks_df['tempo'])

# Round Loudness - user
selected_tracks_df['loudness'] = round(selected_tracks_df['loudness'], 2)

# Display user tracks
show_cols = ['artist_name', 'track_name', 'genre', 'duration', 'tempo', 'mode', 'key', 'loudness']
st.write(selected_tracks_df[show_cols])

# Check if track selected
if len(selected_tracks_df) < 1:
    # Divider
    st.write('---')   

    # Display instructions - if no tracks selected
    st.markdown("<p style='text-align: center; color: grey;'><<<</body>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: grey;'><<<</body>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: grey;'><<<</body>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: grey;'>[select and artist and track for recommendations]</body>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: grey;'><<<</body>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: grey;'><<<</body>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: grey;'><<<</body>", unsafe_allow_html=True)
else:

    ### GEOCORRECT ### ### ###
    g3ocorrected = b3.G3OCORRECT(selected_tracks_df, int(sp_algo), sp_location)

    # Convert duration_ms to minutes and seconds - g3ocorrected
    g3ocorrected['duration_minutes'] = g3ocorrected['duration_ms'] // 60000  # Get minutes
    g3ocorrected['duration_seconds'] = (g3ocorrected['duration_ms'] // 1000) % 60  # Get remaining seconds
    g3ocorrected['duration'] = g3ocorrected['duration_minutes'].astype(str) + ':' + g3ocorrected['duration_seconds'].astype(str).str.zfill(2)  # Combine minutes and seconds into a single column
    g3ocorrected.drop(['duration_minutes', 'duration_seconds'], axis=1, inplace=True)  # Drop temp columns

    # Round Tempo - g3ocorrected
    g3ocorrected['tempo'] = round(g3ocorrected['tempo'])

    # Round Loudness - g3ocorrected
    g3ocorrected['loudness'] = round(g3ocorrected['loudness'], 2)

    # G3ocorrected playlist header
    st.write('_g3ocorrected.')

    # Display g3ocorrected tracks
    st.write(g3ocorrected[show_cols])


    st.write('---') 




    ### MAIN - GEOCORRECTOR - RADAR ### ### ### ### ### ### ### ### ### ### ### ###

    st.write('_radar.') 

    # ACCOUNT FOR DIFFERENT MODE COL TYPES
    df50_modecat = df50.copy()
    df50_modecat['mode'] = np.where(df50_modecat['mode'] == 1, 'Major', 'Minor')

    selected_tracks_df_modecat = selected_tracks_df.copy()
    selected_tracks_df_modecat['mode'] = np.where(selected_tracks_df_modecat['mode'] == 1, 'Major', 'Minor')

    g3ocorrected_modecat = g3ocorrected.copy()
    g3ocorrected_modecat['mode'] = np.where(g3ocorrected_modecat['mode'] == 1, 'Major', 'Minor')

    
    # Get raw nums
    dflab, dfnum = b3.df_numcat(df50_modecat)
    user_dflab, user_dfnum = b3.df_numcat(selected_tracks_df_modecat)
    g3o_dflab, g3o_dfnum = b3.df_numcat(g3ocorrected_modecat)

    # Scale
    # 1. Import the scaler
    from sklearn.preprocessing import StandardScaler 
    # 2. make a scaler
    scaler = StandardScaler()
    # 3. fit the scaler
    scaler.fit(dfnum)
    # 4. transform the data. note we get back a numpy array even if we put in a dataframe
    user_dfnum_scl = scaler.transform(user_dfnum)
    g3o_dfnum_scl = scaler.transform(g3o_dfnum)

    # Add back column titles
    user_dfnum_scl = pd.DataFrame(columns=user_dfnum.columns, data=user_dfnum_scl)
    g3o_dfnum_scl = pd.DataFrame(columns=g3o_dfnum.columns, data=g3o_dfnum_scl)

    # Calculate archetypes
    user_arch_raw = user_dfnum_scl.mean()
    g3o_arch_raw = g3o_dfnum_scl.mean()

    import plotly.graph_objects as go

    # 10 metrics, 9 layers

    # Get the index values as categories
    categories = user_arch_raw.index.tolist()

    # Create traces for each dataset (9 layers)
    trace_user = go.Scatterpolar(r=user_arch_raw.values, theta=categories, fill='toself', name='user', line=dict(color='#28FCC8'))
    trace_g3o = go.Scatterpolar(r=g3o_arch_raw.values, theta=categories, fill='toself', name='g3ocorrected', line=dict(color='#28fc64'))

    # Create figure and add traces
    fig = go.Figure()
    fig.add_trace(trace_user)
    fig.add_trace(trace_g3o)

    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                #range=[0, 1]  # Adjust the range based on your data
            )),
        showlegend=True,
        width=600,  # Set the width of the plot
        height=600,  # Set the height of the plot
        paper_bgcolor='rgba(0,0,0,0)',  # Set the background color to fully transparent
        polar_bgcolor='rgba(0,0,0,0)',  # Set the polar background color to fully transparent
        polar_radialaxis=dict(
            visible=True,
            tickfont=dict(color='lightgrey'),  # Set tick font color to light grey
            gridcolor='grey'  # Set grid color to light grey
        ),
        polar_angularaxis=dict(
            visible=True,
            gridcolor='grey',
            tickfont=dict(color='grey')  # Set tick font color to light grey
        ),
        legend=dict(
            orientation='h',  # Horizontal orientation
            x=0.5,  # Center horizontally
            y=-0.15,  # Place the legend below the plot
            bgcolor='rgba(0,0,0,0)',  # Set the legend background color to fully transparent
            yanchor="top",
            xanchor="center"
        )
    )

    # Plot!
    st.plotly_chart(fig, use_container_width=True)




st.write('---')   




### MAIN - INSIGHTS - BOXPLOTS ### ### ### ### ### ### ### ### ### ### ### ###

st.write("_insights.")

# Sound profile list
sound_profile_list = [london_snd, manchester_snd, nyc_snd, la_snd, kingston_snd, ibiza_snd, berlin_snd] 

# Define the colors for each plot
colors = ['#28fc64', '#28fc64', '#28fc64', '#28fc64', '#28fc64', '#28fc64', '#28fc64']

# Define location titles
sound_profile_strings = ['London', 'Manchester', 'NYC', 'LA', 'Kingston', 'Ibiza', 'Berlin']

# Create a list to store box plot traces
box_traces = []

# Iterate over each dataframe and its corresponding color and location title
for sound, color, title in zip(sound_profile_list, colors, sound_profile_strings):
    # Create box plot trace
    box_trace = go.Box(
        y=sound[metric],
        name=title,
        marker=dict(color=color),
        boxmean=True,  # Display mean line inside the box
        hoverinfo='y+name'  # Show y value and trace name on hover
    )
    box_traces.append(box_trace)

# Create global box plot trace
box_trace = go.Box(
    y=df50[metric],
    name='Global',
    marker_color="#28FCC8",
    boxmean=True,  # Display mean line inside the box
    hoverinfo='y+name'  # Show y value and trace name on hover
)
box_traces.append(box_trace)

# Create layout
layout = go.Layout(
    xaxis=dict(title='location'),
    yaxis=dict(title=metric),
    showlegend=False,
    height=650,
    margin=dict(t=0),
    plot_bgcolor='rgba(0,0,0,0)'
)

# Create figure
fig = go.Figure(data=box_traces, layout=layout)

# Plot!
st.plotly_chart(fig, use_container_width=True)



### MAIN - INSIGHTS - GENRE SUNBURST ### ### ### ### ### ### ### ### ### ### ### ###




st.write('---')   



### MAIN - INSIGHTS - SOUND PROFILE LOCATION MAP ### ### ### ### ### ### ### ### ### ### ### ###

st.write('_locations.') 

data = {
    'sound_profile': ['london', 'manchester', 'nyc', 'la', 'kingston', 'ibiza', 'berlin'],
    'lat': [51.509865, 53.4808, 39.0200, 52.5200, 18.0179, 40.7128, 34.0549],
    'lon': [-0.118092, -2.2426, 1.4821, 13.4050, -76.8099, -74.0060, -118.2426]
}

map = pd.DataFrame(data)

st.map(map, color='#28fc64')   
     
st.write('---')   





### MAIN - G3TSIMILAR ### ### ### ### ### ### ### ### ### ### ### ###

#st.write('_g3tsimilar.') 



#st.write('---')   


### NOTES ### ### ### ### ### ### ### ### ### ### ### ###

st.write('_notes.')   
st.write('- *algorithm 1 - top ranked cosine similarity vs sound profile [primary - best performance]*.') 
st.write('- *algorithm 2 - top ranked cosine similarity vs upsampled sound profile (random forest) [experimental]*.') 
st.write('- *algorithm 3 - top ranked cosine similarity vs upsampled sound profile (cosine similarity) [experimental]*.') 
st.write('- *radar chart reflects similarity of playlists (first track is direct track comparision)*.') 
st.write('- *questions? message b3tascape@gmail.com*.') 

#######################################################################################################################################
#######################################################################################################################################

### NOTES

### LAUNCHING THE APP ON THE LOCAL MACHINE ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
### 1. Save your *.py file (the file and the dataset should be in the same folder)
### 2. Open git bash (Windows) or Terminal (MAC) and navigate (cd) to the folder containing the *.py and *.csv files
### 3. Execute... streamlit run <name_of_file.py>
### 4. The app will launch in your browser. A 'Rerun' button will appear every time you SAVE an update in the *.py file


### GENERAL ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
### Create a title
    #### st.title("Kickoff - Live coding an app")
### You can also use markdown syntax (here we use the ### symbols at the front to make a heading)
    #### st.write('### The best morning kickoff ever, but the last one :(')
### To position text and color, you can use html syntax
    ### st.markdown("<h3 style='text-align: center; color: blue;'>Data: NYC Bike Sharing Dataset</h1>", unsafe_allow_html=True)


### MODEL INFERENCE ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
#### Models can be imported using *.pkl files (or similar) so predictions, classifications etc can be done within the app using previously optimized models
#### Automating processes and handling real-time data

#st.subheader("Using pretrained models with user input")

# A. Load the model using joblib

#model = joblib.load("sentiment_pipeline.pkl")
# B. Set up input field

#text = st.text_input('Enter your review text below', 'Best. Restaurant. Ever.')
# C. Use the model to predict sentiment & write result
#prediction = model.predict([text])
#    st.write("The model predicts this is a positive review")
#if prediction == 1:
#else:
#    st.write("The model predicts this is a negative review")



#######################################################################################################################################
### Streamlit Advantages and Disadvantages
    
#st.subheader("Streamlit Advantages and Disadvantages")
#st.write('**Advantages**')
#st.write(' - Easy, Intuitive, Pythonic')


