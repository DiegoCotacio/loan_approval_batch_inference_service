import os
import sys
import dill

import pandas as pd
import numpy as np
import pickle
from google.cloud import bigquery
from google.oauth2 import service_account


from .exception import CustomException
from .logger import logging
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV



# Funcion para guardar objetos
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok= True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)
    


#funcion para cargar objetos
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)
    


def connect_to_bigquery():

    # Load BigQuery credentials from the secret
    #credentials_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')

    # Load BigQuery credentials from the service_account.json file
    #credentials = service_account.Credentials.from_service_account_info(credentials_json)

    credentials = service_account.Credentials.from_service_account_file('src/cred/protean-fabric-386717-d6a21dd66382.json')

    # Connect to the BigQuery API using the credentials
    client = bigquery.Client(credentials=credentials)
    
    return client