from sklearn.base import  ClassifierMixin
import logging
import pandas as pd
import numpy as np
from prefect import task, flow
import mlflow
import os
import sys
import joblib
import pickle
import optuna
from lightgbm import LGBMClassifier
from urllib.parse import urlparse
#--------- Custom packages
from src.components.data_preprocessing import DataTransformation
from src.components.model_evaluator import Evaluation
from src.components.data_ingestion import DataIngestion
from src.components.model_optimizer import Hyperparameter_Optimization
from src.utils import save_object
from src.exception import CustomException
from src.logger import logging


# -------------------- DATA INGESTION STEP 

@task
def ingest_data() -> pd.DataFrame:
    """
    Args: None
    Returns: df: pd.DataFrame
    """
    try:
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()

        return train_data, test_data
    
    except Exception as e:
            raise CustomException(e, sys)
    

# --------------------DATA CLEANING SETP


@task
def transform_data(train_data: pd.DataFrame, test_data: pd.DataFrame):
    """ Es una clase que preprocesa y divide los datos """
    try:
        preprocessor = DataTransformation()  # Crear una instancia de PreprocessData

        train_array, test_array,_= preprocessor.initiate_data_transformation(train_data, test_data)

        return train_array, test_array
    
    except Exception as e:
            raise CustomException(e, sys)

#-------------------------- MODEL TRAINING STEP


@task
def train_model(train_array, test_array):
    logging.info("Started model training")
    try:
         logging.info("split training and test input data")
         X_train, y_train, X_test, y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1])
         
         logging.info("Started training XGBoost model.")
         
         hy_opt = Hyperparameter_Optimization(X_train, y_train, X_test, y_test)
         
         study = optuna.create_study(direction = "maximize")
         study.optimize(hy_opt.classifier_optimizer, n_trials = 10)
         trial = study.best_trial
         
         n_estimators = trial.params["n_estimators"]
         learning_rate = trial.params["learning_rate"]
         max_depth = trial.params["max_depth"]
         
         model = LGBMClassifier(
             n_estimators = n_estimators,
             learning_rate = learning_rate,
             max_depth = max_depth,
             )
         
         model.fit(X_train, y_train)

         trained_model_file_path = os.path.join("artifacts", "model.pkl")
         save_object(
                file_path = trained_model_file_path,
                obj = model
            )
         
         return model
    
    except Exception as e:
            raise CustomException(e, sys)

#--------------------- MODEL EVALUATION STEP

@task
def evaluation(model: ClassifierMixin, test_array:np.ndarray):
    
    "Args: model, x_test, y_test  and Returns: r2_score and rmse"

    try:
        logging.info("setting test data for evaluation")
        X_test, y_test =(
                test_array[:,:-1],
                test_array[:,-1])
        
        prediction = model.predict(X_test)
        evaluation = Evaluation()

        accuracy = evaluation.accuracy(y_test, prediction)
        precision = evaluation.precision(y_test, prediction)
        recall = evaluation.recall(y_test, prediction) 
        f1_score = evaluation.f1_score(y_test, prediction)

        return accuracy, precision, recall, f1_score
    
    except Exception as e:
            raise CustomException(e, sys)

#- ------------------------- TRAINING PIPELINE
@flow(name="Training Pipeline for Loan Approbal prediction model")
def train_pipeline():
     train_data, test_data = ingest_data()
     train_array, test_array = transform_data(train_data, test_data)
     model = train_model(train_array, test_array)
     accuracy, precision, recall, f1_score = evaluation(model, test_array)
     # Imprimir las métricas en la consola
     print(f"Accuracy: {accuracy}")
     print(f"Precision: {precision}")
     print(f"Recall: {recall}")
     print(f"F1 Score: {f1_score}")


if __name__ == "__main__":
 train_pipeline()