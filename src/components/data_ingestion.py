import pandas as pd
import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.components.data_preprocessing import DataTransformation, DataTransformationConfig
#from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', "data.csv")
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")



class DataIngestion:


    def __init__(self):
        self.ingestion_config= DataIngestionConfig()


    def initiate_data_ingestion(self):

        logging.info("Entered the data ingestion component")
        
        try:
            loans_df = pd.read_csv('data/loans.csv')
            applicants_df = pd.read_csv('data/applicants.csv')

            df = pd.merge(loans_df, applicants_df, on='id', how='left')

            df.drop([ 'title', 'grade', 'address','emp_title', 'emp_length', 'id', 'issue_d','earliest_cr_line'], axis = 1, inplace = True)

            def format_dtypes(df):
                 
                 # Convertir las columnas específicas a tipos de datos deseados
                 df['term'] = df['term'].astype(str)
                 df['sub_grade'] = df['sub_grade'].astype(str)
                 df['purpose'] = df['purpose'].astype(str)
                 df['home_ownership'] = df['home_ownership'].astype(str)
                 df['verification_status'] = df['verification_status'].astype(str)
                 df['initial_list_status'] = df['initial_list_status'].astype(str)
                 df['application_type'] = df['application_type'].astype(str)
                 df['loan_status'] = df['loan_status'].astype(str)  # Mantener el mismo mapeo de 'loan_status' que en la función de referencia
                 
                 # Convertir columnas numéricas específicas a float
                 numeric_columns = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti', 'open_acc',
                       'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'mort_acc', 'pub_rec_bankruptcies']
                 df[numeric_columns] = df[numeric_columns].astype(float)
                 
                 # Mapeo para 'churn' (loan_status) y conversión a str
                 mapping = {'loan_status': {'Fully Paid': 1, 'Charged Off': 0}}
                 df.replace(mapping, inplace=True)
                 df['loan_status'] = df['loan_status'].astype(int)

                 # Mover 'loan_status' al final del DataFrame
                 loan_status_column = df.pop('loan_status')
                 df['loan_status'] = loan_status_column

                 return df
            
            df = format_dtypes(df)


            logging.info("Load the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok= True)
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header= True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size= 0.2, random_state= 42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header = True)


            logging.info("Ingestion is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
        
  # test local component
#if __name__=="__main__":
 #   obj = DataIngestion()
  #  train_data, test_data = obj.initiate_data_ingestion()

    #data_transformation = DataTransformation()
    #train_array, test_array,_= data_transformation.initiate_data_transformation(train_data, test_data)
     
    #modeltrainer = ModelTrainer()
    #print(modeltrainer.initiate_model_trainer(train_array, test_array)) #return r2_score
