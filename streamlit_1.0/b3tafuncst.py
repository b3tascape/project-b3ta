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


### FUNCTION 01 --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 

def df_snapshot_vc(df):
    
    col_list = list(df.columns)
    for col in col_list:
        print(f"{col.upper()}:\n\n{df[col].value_counts()}\n\n ============================================================= \n")
        
#DOCUMENTATION;
#- Takes a single DataFrame as an argument.
#- Prints a value_count() snapshot of each column in a dataframe. 



### FUNCTION 02 --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 

def df_numcat(df):
    
    df_cat = df.select_dtypes(['object']).copy()
    df_num = df.select_dtypes(['int64','float64', 'datetime64[ns]']).copy()    
    return df_cat, df_num

#DOCUMENTATION;
#- Takes a single DataFrame as an argument.
#- Returns two DataFrames; A copy of the categorical columns (df_cat) and a copy of the numerical columns (df_num)
#- Simply assign output to 2x variables e.g.; df_cat, df_num = df_numcat(df)



### FUNCTION 03 --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 

def df_numstats(df):
    import numpy as np

    col_list = list(df.columns)
    for col in col_list:
        print(col.upper())
        print(f"MEAN: {round(np.mean(df[col]),2)}")
        print(f"MEDIAN: {round(np.median(df[col]),2)}")
        print(f"RANGE: {round(np.ptp(df[col]),2)}")
        print(f"IQR: {round(scipy.stats.iqr(df[col]),2)}")
        print(f"STANDARDDEVIATION: {round(np.std(df[col]),2)}")
        print(f"MAX: {round(np.max(df[col]),2)}")
        print(f"MIN: {round(np.min(df[col]),2)}")
        print("\n\n ============================================================= \n")

#DOCUMENTATION;
#- Takes a single DataFrame as an argument (Numeric variables only).
#- Prints key stats metrics for each column; Mean, Median, Range, IQR, Standard Deviation, Max and Min



### FUNCTION 04 --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 

def df_histoboxme(df):
    col_list = list(df.columns)
    fig, axes = plt.subplots(len(col_list), 2, figsize=(15, 5*len(col_list)))

    for i, col in enumerate(col_list):
        sns.histplot(data=df, x=col, ax=axes[i, 0], color='#28fc64')
        axes[i, 0].set_title(col + " - Histogram")
        axes[i, 0].set_xlabel(col)

        sns.boxplot(data=df, x=col, ax=axes[i, 1], color='#28fc64')
        axes[i, 1].set_title(col + " - Boxplot")
        axes[i, 1].set_xlabel(col)

    plt.tight_layout()
    plt.show()
    
#DOCUMENTATION;
#- Creates a histogram and box plot for a df of numeric values



### FUNCTION 05 --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 

def DROP_DUPLICATES(sp_list, sp_names_list, sp_drop_priority=None):
    """
    Drop duplicates from each DataFrame in sp_list and print the duplicated value counts for each DataFrame. Note this process is DESTRUCTIVE, duplicates will be dropped from the original DataFrames.

    Args:
        sp_list (list): List of DataFrames.
        sp_names_list (list): List of names corresponding to each DataFrame.
        sp_drop_priority (string, optional): Name of sound_profile tag to prioritise dropping.
            Only use if confident all points to be dropped have that sound_profile tag. Default is None.

    Returns:
        None
    """
    # Validate input parameters
    if len(sp_list) != len(sp_names_list):
        raise ValueError("Length of sp_list and sp_names_list must be the same.")

    # Loop through DataFrames
    for sp, sp_name in zip(sp_list, sp_names_list):

        # Drop duplicates
        dupe_indices = sp[sp['track_id'].duplicated()].index
        if sp_drop_priority:
            # Filter duplicates and priority sound_profile
            dupe_filter = sp['track_id'].duplicated(keep=False)
            sp_unknown_filter = sp['sound_profile'] == sp_drop_priority
            dupe_indices = sp[dupe_filter & sp_unknown_filter].index
            # Drop duplicates
            sp.drop(index=dupe_indices, inplace=True)
        else:
            # Filter duplicates and priority sound_profile
            
            # Drop duplicates
            sp.drop(index=dupe_indices, inplace=True)

        # Print duplicated value counts
        print(f"{sp_name} duplicates:")
        print(f"{sp.duplicated().value_counts()}")
        print("---")



### FUNCTION 06 --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 

def MERGE2MAIN(df, new_sp):
    dfc = df.copy()
    index2drop = []
    for index, row in dfc.iterrows():  # Iterate over rows
        if row['track_id'] in new_sp['track_id'].values:  # Check if track_id exists in new_sp
            index2drop.append(index)
    print(f"Number of duplicates: {len(index2drop)}")
    print(f"Indices to drop: {index2drop}")
    dfc.drop(index=index2drop, inplace=True)  # Use inplace=True to drop rows from dfc
    merged = pd.concat([dfc, new_sp])
    return merged

#DOCUMENTATION;
#- Merges an updated sound profile to the main database, dropping any duplicates
#- Takes two arguments, main df and new sound profile, both must have same columns, types etc.



############ G3O-CORRECTER ############

#######################################################################################################################################
#######################################################################################################################################

### IMPORTS

# full df - raw
df50 = pd.read_csv('project-b3ta/streamlit_1.0/06-b3_df50-sl.csv', index_col=0)
user_raw_df = pd.read_csv('project-b3ta/streamlit_1.0/04-b3-user_raw_df.csv', index_col=0)

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
london_upsamp_test_rf = pd.read_csv('project-b3ta/streamlit_1.0/london_upsamp_test_rf-mergedv1.csv', index_col=0)
manchester_upsamp_test_rf = pd.read_csv('project-b3ta/streamlit_1.0/manchester_upsamp_test_rf-mergedv1.csv', index_col=0)
ibiza_upsamp_test_rf = pd.read_csv('project-b3ta/streamlit_1.0/ibiza_upsamp_test_rf-mergedv1.csv', index_col=0)
berlin_upsamp_test_rf = pd.read_csv('project-b3ta/streamlit_1.0/berlin_upsamp_test_rf-mergedv1.csv', index_col=0)
kingston_upsamp_test_rf = pd.read_csv('project-b3ta/streamlit_1.0/kingston_upsamp_test_rf-mergedv1.csv', index_col=0)
nyc_upsamp_test_rf = pd.read_csv('project-b3ta/streamlit_1.0/nyc_upsamp_test_rf-mergedv1.csv', index_col=0)
la_upsamp_test_rf = pd.read_csv('project-b3ta/streamlit_1.0/la_upsamp_test_rf-mergedv1.csv', index_col=0)

# upsampled_cs
london_upsamp_test_cs = pd.read_csv('project-b3ta/streamlit_1.0/london_upsamp_test_cs-mergedv1.csv', index_col=0)
manchester_upsamp_test_cs = pd.read_csv('project-b3ta/streamlit_1.0/manchester_upsamp_test_cs-mergedv1.csv', index_col=0)
ibiza_upsamp_test_cs = pd.read_csv('project-b3ta/streamlit_1.0/ibiza_upsamp_test_cs-mergedv1.csv', index_col=0)
berlin_upsamp_test_cs = pd.read_csv('project-b3ta/streamlit_1.0/berlin_upsamp_test_cs-mergedv1.csv', index_col=0)
kingston_upsamp_test_cs = pd.read_csv('project-b3ta/streamlit_1.0/kingston_upsamp_test_cs-mergedv1.csv', index_col=0)
nyc_upsamp_test_cs = pd.read_csv('project-b3ta/streamlit_1.0/nyc_upsamp_test_cs-mergedv1.csv', index_col=0)
la_upsamp_test_cs = pd.read_csv('project-b3ta/streamlit_1.0/la_upsamp_test_cs-mergedv1.csv', index_col=0)


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
    print(f"PLAYLIST SHAPE: {playlist.shape}")
    print(f"PLAYLIST NUM: {playlist_num.shape}")
    print(f"PLAYLIST LAB: {playlist_labels.shape}")

    # 2a) Scale target playlist numeric - import the scaler
    from sklearn.preprocessing import StandardScaler

    # 2b) Scale target playlist numeric - make a scaler
    scaler = StandardScaler()

    # 2c) Scale target playlist numeric - fit the scaler
    df_labels, df_num = b3.df_numcat(df50)
    scaler.fit(df_num)
    print(f"DF50 SHAPE: {df50.shape}")
    print(f"DF50 NUM: {df_num.shape}")
    print(f"DF50 LAB: {df_labels.shape}")

    # 2d) Scale target playlist numeric - transform the data. note we get back a numpy array even if we put in a dataframe
    playlist_num_scl = scaler.transform(playlist_num)

    # 2e) Scale target playlist numeric - convert to df and add back columns
    playlist_num_scl = pd.DataFrame(columns=playlist_num.columns, data=playlist_num_scl)


    ### GEOCORRECT - IF LOCATION VALID

    # 1) Split target playlist into lables and numeric
    temp_labels, temp_num = b3.df_numcat(temp)
    print(f"DF50 SHAPE: {temp.shape}")
    print(f"DF50 NUM: {temp_num.shape}")
    print(f"DF50 LAB: {temp_labels.shape}")

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
            ### top 20 print
            top_20 = ['track_id', 'artist_name', 'track_name']
            print(temp.loc[temp_csdf.sort_values(by=0, ascending=False).index,:][top_20])
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

### FUNCTION 002 --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 

def G3TSIMILAR(track):

    ### CREATE EMPTY GEO-CORRECTED DF
    similar_tracks = pd.DataFrame()

    ### PREP USER PLAYLIST
    # 1) Split target playlist into labels and numeric
    track_labels, track_num = b3.df_numcat(track)

    # 2a) Scale target playlist numeric - import the scaler
    from sklearn.preprocessing import StandardScaler

    # 2b) Scale target playlist numeric - make a scaler
    scaler = StandardScaler()

    # 2c) Scale target playlist numeric - fit the scaler
    df_labels, df_num = b3.df_numcat(df50)
    scaler.fit(df_num)

    # 2d) Scale target playlist numeric - transform the data. note we get back a numpy array even if we put in a dataframe
    track_num_scl = scaler.transform(track_num)

    # 2e) Scale target playlist numeric - convert to df and add back columns
    track_num_scl = pd.DataFrame(columns=track_num.columns, data=track_num_scl)

      

    ### G3TSIMILAR

    # 0) Get global
    temp = df.copy()
    temp.reset_index(drop=True, inplace=True)  

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

    # COSINE COMPARISON

    #### Create temp cosine similarity df
    temp_csdf = pd.DataFrame(cosine_similarity(track_num_scl.loc[0,:].values.reshape(1, -1), temp_num_scl)).T

    #### Get index of top 10 results
    # IF GLOBAL
    if location.lower() == 'global':
        temp_index = temp_csdf.sort_values(by=0, ascending=False).index[1]
        ### top 20 print
        top_20 = ['track_id', 'artist_name', 'track_name']
        print(temp.loc[temp_csdf.sort_values(by=0, ascending=False).index,:][top_20])
    else:
        temp_index = temp_csdf.sort_values(by=0, ascending=False).index[1:11]

    #### Grab track from temp df and add to similar_tracks playlist
    similar_tracks = pd.concat([similar_tracks, temp.loc[[temp_index]]][0:10])

    #### Drop track from temp dfs so no replication in final playlist
    temp.drop(index=temp_index, inplace=True)
    temp_num_scl.drop(index=temp_index, inplace=True)
    #temp.reset_index(drop=True, inplace=True)
    #temp_num_scl.reset_index(drop=True, inplace=True) 

    return similar_tracks

#DOCUMENTATION;
#- Takes a single DataFrame (raw, 17 columns) and a location as an argument.
#- Returns a geo-corrected playlist (e.g. playlist_geocorrected_for_london = df_geocorrect1(user_raw_df, 'london') )




############ NOTES ############


### COLOUR GUIDE --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 


# colour options - primary
    #28fc81
    #28fc64 <-
    #1DFD54

# colour options - secondary
    #28FCC8 - green/blue secondary
    #D4FB79 - green
    #FF7E79 - red
    #7A81FF - blue

# %%
