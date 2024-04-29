### import libraries
import numpy as np
import pandas as pd
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


# other
import numbers
import requests

# b3ta functions
import b3tafuncst as b3


############ GENERAL ############


### FUNCTION 001 --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 

def df_numcat(df):
    
    df_cat = df.select_dtypes(['object']).copy()
    df_num = df.select_dtypes(['int64','float64', 'datetime64[ns]']).copy()    
    return df_cat, df_num

#DOCUMENTATION;
#- Takes a single DataFrame as an argument.
#- Returns two DataFrames; A copy of the categorical columns (df_cat) and a copy of the numerical columns (df_num)
#- Simply assign output to 2x variables e.g.; df_cat, df_num = df_numcat(df)




############ G3O-CORRECTER ############

#######################################################################################################################################
#######################################################################################################################################

### IMPORTS

# full df - raw
df50 = pd.read_csv('streamlit_1.0/06-b3_df50-sl.csv', index_col=0)
user_raw_df = pd.read_csv('streamlit_1.0/04-b3-user_raw_df.csv', index_col=0)

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
london_upsamp_test_rf = pd.read_csv('streamlit_1.0/london_upsamp_test_rf-mergedv1.csv', index_col=0)
manchester_upsamp_test_rf = pd.read_csv('streamlit_1.0/manchester_upsamp_test_rf-mergedv1.csv', index_col=0)
ibiza_upsamp_test_rf = pd.read_csv('streamlit_1.0/ibiza_upsamp_test_rf-mergedv1.csv', index_col=0)
berlin_upsamp_test_rf = pd.read_csv('streamlit_1.0/berlin_upsamp_test_rf-mergedv1.csv', index_col=0)
kingston_upsamp_test_rf = pd.read_csv('streamlit_1.0/kingston_upsamp_test_rf-mergedv1.csv', index_col=0)
nyc_upsamp_test_rf = pd.read_csv('streamlit_1.0/nyc_upsamp_test_rf-mergedv1.csv', index_col=0)
la_upsamp_test_rf = pd.read_csv('streamlit_1.0/la_upsamp_test_rf-mergedv1.csv', index_col=0)

# upsampled_cs
london_upsamp_test_cs = pd.read_csv('streamlit_1.0/london_upsamp_test_cs-mergedv1.csv', index_col=0)
manchester_upsamp_test_cs = pd.read_csv('streamlit_1.0/manchester_upsamp_test_cs-mergedv1.csv', index_col=0)
ibiza_upsamp_test_cs = pd.read_csv('streamlit_1.0/ibiza_upsamp_test_cs-mergedv1.csv', index_col=0)
berlin_upsamp_test_cs = pd.read_csv('streamlit_1.0/berlin_upsamp_test_cs-mergedv1.csv', index_col=0)
kingston_upsamp_test_cs = pd.read_csv('streamlit_1.0/kingston_upsamp_test_cs-mergedv1.csv', index_col=0)
nyc_upsamp_test_cs = pd.read_csv('streamlit_1.0/nyc_upsamp_test_cs-mergedv1.csv', index_col=0)
la_upsamp_test_cs = pd.read_csv('streamlit_1.0/la_upsamp_test_cs-mergedv1.csv', index_col=0)


### FUNCTION 001 --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 

def G3OCORRECT(playlist, algorithm, location):

    ### APPLY ALGORITHM
    if algorithm == 1:
        # IF LONDON
        if location.lower() == 'london':
            temp = london_snd.copy()
            temp.reset_index(drop=True, inplace=True)
            
        # IF MANCHESTER
        elif location.lower() == 'manchester':
            temp = manchester_snd.copy()
            temp.reset_index(drop=True, inplace=True)
    
        # IF IBIZA
        elif location.lower() == 'ibiza':
            temp = ibiza_snd.copy()
            temp.reset_index(drop=True, inplace=True)   

        # IF BERLIN
        elif location.lower() == 'berlin':
            temp = berlin_snd.copy()
            temp.reset_index(drop=True, inplace=True)    

        # IF KINGSTON
        elif location.lower() == 'kingston':
            temp = kingston_snd.copy()
            temp.reset_index(drop=True, inplace=True)

        # IF NYC
        elif location.lower() == 'nyc':
            temp = nyc_snd.copy()
            temp.reset_index(drop=True, inplace=True)

        # IF LA
        elif location.lower() == 'la':
            temp = la_snd.copy()
            temp.reset_index(drop=True, inplace=True)

        # IF GLOBAL
        elif location.lower() == 'global':
            temp = df50.copy()
            temp.reset_index(drop=True, inplace=True)        

        # ELSE ENTER VALID LOCATION
        else:
            print("ENTER VALID LOCATION: Input one of the following 'London', 'Manchester', 'Ibiza', 'Berlin', 'Kingston', 'NYC', 'LA', 'Global'")
            
    elif algorithm == 2:
        # IF LONDON
        if location.lower() == 'london':
            temp = london_upsamp_test_rf.copy()
            temp.reset_index(drop=True, inplace=True)

        # IF MANCHESTER
        elif location.lower() == 'manchester':
            temp = manchester_upsamp_test_rf.copy()
            temp.reset_index(drop=True, inplace=True)
    
        # IF IBIZA
        elif location.lower() == 'ibiza':
            temp = ibiza_upsamp_test_rf.copy()
            temp.reset_index(drop=True, inplace=True)   

        # IF BERLIN
        elif location.lower() == 'berlin':
            temp = berlin_upsamp_test_rf.copy()
            temp.reset_index(drop=True, inplace=True)    

        # IF KINGSTON
        elif location.lower() == 'kingston':
            temp = kingston_upsamp_test_rf.copy()
            temp.reset_index(drop=True, inplace=True)

        # IF NYC
        elif location.lower() == 'nyc':
            temp = nyc_upsamp_test_rf.copy()
            temp.reset_index(drop=True, inplace=True)

        # IF LA
        elif location.lower() == 'la':
            temp = la_upsamp_test_rf.copy()
            temp.reset_index(drop=True, inplace=True)

        # IF GLOBAL
        elif location.lower() == 'global':
            temp = df50.copy()
            temp.reset_index(drop=True, inplace=True)        

        # ELSE ENTER VALID LOCATION
        else:
            print("ENTER VALID LOCATION: Input one of the following 'London', 'Manchester', 'Ibiza', 'Berlin', 'Kingston', 'NYC', 'LA', 'Global'")

    elif algorithm == 3:

        # IF LONDON
        if location.lower() == 'london':
            temp = london_upsamp_test_cs.copy()
            temp.reset_index(drop=True, inplace=True)

        # IF MANCHESTER
        elif location.lower() == 'manchester':
            temp = manchester_upsamp_test_cs.copy()
            
    
        # IF IBIZA
        elif location.lower() == 'ibiza':
            temp = ibiza_upsamp_test_cs.copy()
            temp.reset_index(drop=True, inplace=True)   

        # IF BERLIN
        elif location.lower() == 'berlin':
            temp = berlin_upsamp_test_cs.copy()
            temp.reset_index(drop=True, inplace=True)    

        # IF KINGSTON
        elif location.lower() == 'kingston':
            temp = kingston_upsamp_test_cs.copy()
            temp.reset_index(drop=True, inplace=True)

        # IF NYC
        elif location.lower() == 'nyc':
            temp = nyc_upsamp_test_cs.copy()
            temp.reset_index(drop=True, inplace=True)

        # IF LA
        elif location.lower() == 'la':
            temp = la_upsamp_test_cs.copy()
            temp.reset_index(drop=True, inplace=True)

        # IF GLOBAL
        elif location.lower() == 'global':
            temp = df50.copy()
            temp.reset_index(drop=True, inplace=True)      

        # ELSE ENTER VALID LOCATION
        else:
            print("ENTER VALID LOCATION: Input one of the following 'London', 'Manchester', 'Ibiza', 'Berlin', 'Kingston', 'NYC', 'LA', 'Global'")

    else:
        print("ENTER A RECOMMENDATION ALGORITHM 1, 2 or 3")

    
    # ACCOUNT FOR DIFFERENT MODE COL TYPES
    df50.reset_index(drop=True, inplace=True)
    value = df50['mode'].loc[0]
    if isinstance(value, numbers.Number):
        df50['mode'] = np.where(df50['mode'] == 1, 'Major', 'Minor')

    playlist.reset_index(drop=True, inplace=True)
    value = playlist['mode'].loc[0]
    if isinstance(value, numbers.Number):
        playlist['mode'] = np.where(playlist['mode'] == 1, 'Major', 'Minor')

    temp.reset_index(drop=True, inplace=True)
    value = temp['mode'].loc[0]
    if isinstance(value, numbers.Number):
        temp['mode'] = np.where(temp['mode'] == 1, 'Major', 'Minor')
        
    ### PREP USER PLAYLIST
    # 1) Split target playlist into labels and numeric
    playlist_labels, playlist_num = b3.df_numcat(playlist)

    # 2a) Scale target playlist numeric - import the scaler
    from sklearn.preprocessing import StandardScaler

    # 2b) Scale target playlist numeric - make a scaler
    scaler = StandardScaler()

    # 2c) Scale target playlist numeric - fit the scaler
    df_labels, df_num = b3.df_numcat(df50)
    scaler.fit(df_num)

    # 2d) Scale target playlist numeric - transform the data. note we get back a numpy array even if we put in a dataframe
    playlist_num_scl = scaler.transform(playlist_num)

    # 2e) Scale target playlist numeric - convert to df and add back columns
    playlist_num_scl = pd.DataFrame(columns=playlist_num.columns, data=playlist_num_scl)


    ### GEOCORRECT - IF LOCATION VALID

    # 1) Split target playlist into lables and numeric
    temp_labels, temp_num = b3.df_numcat(temp)

    # 2a) Scale target playlist numeric - import the scaler
    from sklearn.preprocessing import StandardScaler

    # 2b) Scale target playlist numeric - make a scaler
    scaler = StandardScaler()

    # 2c) Scale target playlist numeric - fit the scaler
    scaler.fit(df_num)

    # 2d) Scale target playlist numeric - transform the data. note we get back a numpy array even if we put in a dataframe
    temp_num_scl = scaler.transform(temp_num)

    # 2e) Scale target playlist numeric - convert to df and add back columns
    temp_num_scl = pd.DataFrame(columns=temp_num.columns, data=temp_num_scl)

    ### CREATE EMPTY GEO-CORRECTED DF
    geocorrected = pd.DataFrame()

    # COSINE COMPARISON
    for i in range(playlist_num_scl.shape[0]):

        #### Create temp cosine similarity df
        temp_csdf = pd.DataFrame(cosine_similarity(playlist_num_scl.loc[i,:].values.reshape(1, -1), temp_num_scl)).T

        #### Get index of top result
        # IF GLOBAL
        if location.lower() == 'global':
            temp_index = temp_csdf.sort_values(by=0, ascending=False).index[1]
        else:
            temp_index = temp_csdf.sort_values(by=0, ascending=False).index[0]

        #### Grab track from temp df and add to geocorrected playlist
        geocorrected = pd.concat([geocorrected, temp.loc[[temp_index]]][0:10])

        #### Drop track from temp dfs so no replication in final playlist
        temp.drop(index=temp_index, inplace=True)
        temp_num_scl.drop(index=temp_index, inplace=True)
        temp.reset_index(drop=True, inplace=True)
        temp_num_scl.reset_index(drop=True, inplace=True) 

    return geocorrected

#DOCUMENTATION;
#- Takes a single DataFrame (raw, 17 columns) and a location as an argument.
#- Returns a geo-corrected playlist (e.g. playlist_geocorrected_for_london = df_geocorrect1(user_raw_df, 'london') )




############ NOTES ############

