"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def remove_outliers(df, columns, n_std):
    for col in columns:
    
        # setup variables for outlier consideration
        check_kurtosis = df[col].kurtosis()
        mean = df[col].mean()
        sd = df[col].std()
    
        # apply kurtosis check 
        if check_kurtosis > 3:
            #df = df[(df[col] <= mean + (n_std*sd))]
            df = df[(df[col] >= mean - (n_std*sd)) & (df[col] <= mean + (n_std*sd))] 

        # remove outliers
        #df = df[(df[col] >= mean - (n_std*sd)) & (df[col] <= mean + (n_std*sd))] 
        #df = df[(df[col] <= mean + (n_std*sd))]

    return df

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    feature_vector_df["Valencia_pressure"].fillna(1012, inplace=True)
    feature_vector_df["Valencia_wind_deg"] = feature_vector_df["Valencia_wind_deg"].str.extract("(\d+)")
    
    feature_vector_df["Valencia_wind_deg"] = pd.to_numeric(feature_vector_df["Valencia_wind_deg"])
    
    feature_vector_df["Seville_pressure"] = feature_vector_df["Seville_pressure"].str.extract("(\d+)")
    
    feature_vector_df["Seville_pressure"] = pd.to_numeric(feature_vector_df["Seville_pressure"])

    feature_vector_df["Madrid_temp_range"] = feature_vector_df["Madrid_temp_max"]-feature_vector_df["Madrid_temp_min"]
    feature_vector_df["Seville_temp_range"] = feature_vector_df["Seville_temp_max"]-feature_vector_df["Seville_temp_min"]
    feature_vector_df["Bilbao_temp_range"] = feature_vector_df["Bilbao_temp_max"]-feature_vector_df["Bilbao_temp_min"]
    feature_vector_df["Barcelona_temp_range"] = feature_vector_df["Barcelona_temp_max"]-feature_vector_df["Barcelona_temp_min"]
    feature_vector_df["Valencia_temp_range"] = feature_vector_df["Valencia_temp_max"]-feature_vector_df["Valencia_temp_min"]
    

    feature_vector_df["time"] = pd.to_datetime(feature_vector_df["time"])
    feature_vector_df["Year"] = feature_vector_df["time"].dt.year
    feature_vector_df["Month_Num"] = feature_vector_df["time"].dt.month
    feature_vector_df["Day_Num_of_Year"] = feature_vector_df["time"].dt.day_of_year
    feature_vector_df["Day_Date"] = feature_vector_df["time"].dt.day
    feature_vector_df["Day_of_Week"] = feature_vector_df["time"].dt.day_of_week
    feature_vector_df["Week_Day"] = feature_vector_df["Day_of_Week"].apply(lambda x: 1 if x <= 4 else 0)
    feature_vector_df["Weekend"] = feature_vector_df["Day_of_Week"].apply(lambda x: 1 if x > 4 else 0)
    feature_vector_df["Hour"] = feature_vector_df["time"].dt.hour
    feature_vector_df["AM"] = feature_vector_df["Hour"].apply(lambda x: 1 if x >= 0 and x < 12 else 0)
    feature_vector_df["PM"] = feature_vector_df["Hour"].apply(lambda x: 1 if x >= 12 and x <= 23 else 0)
    feature_vector_df["Day_Month_Start_0"] = feature_vector_df["time"].dt.is_month_start
    feature_vector_df["Day_Month_Start"] = feature_vector_df["Day_Month_Start_0"].apply(lambda x: 1 if x == True else 0)
    feature_vector_df["Day_Month_End_0"] = feature_vector_df["time"].dt.is_month_end
    feature_vector_df["Day_Month_End"] = feature_vector_df["Day_Month_End_0"].apply(lambda x: 1 if x == True else 0)

    feature_vector_df.drop(["Unnamed: 0", "time", "Day_Month_Start_0", "Day_Month_End_0", "Madrid_temp", "Madrid_temp_min", "Seville_temp", "Seville_temp_min", "Seville_temp_max", "Barcelona_temp","Barcelona_temp_min", "Bilbao_temp", "Bilbao_temp_min", "Bilbao_temp_max", "Madrid_temp_max", "Barcelona_temp_max", "Valencia_temp", "Valencia_temp_min", "Valencia_temp_max"], axis = 1, inplace =True)



    # identify numerical columns from dataframe
    column_list = feature_vector_df.iloc[:, (np.where((feature_vector_df.dtypes == np.int64) | (feature_vector_df.dtypes == np.float64)))[0]].columns 
    #column_list = df_train_RO.columns.values

    # call remove_outliers function
    df_train_RO_2 = remove_outliers(feature_vector_df, column_list, 3)

    #predict_vector = feature_vector_df.drop(["Unnamed: 0", "time"], axis=1)

    #predict_vector = feature_vector_df[['Madrid_wind_speed','Bilbao_rain_1h','Valencia_wind_speed']]

    # ------------------------------------------------------------------------

    return df_train_RO_2

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()
