import sys
import os
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    raw_data_path: str = os.path.join('artifacts', "raw.csv")
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    # Custom preprocessing functions for preprocessor building
    def bin_column(self, df, col_name, num_bins=5):
        # Calculate the bin edges to evenly split the numerical column
        bin_edges = pd.qcut(df[col_name], q=num_bins, retbins=True)[1]

        # Define labels for the categorical bins based on bin edges
        bin_labels = [f'{int(bin_edges[i])}-{int(bin_edges[i+1])}' for i in range(num_bins)]

        # Use pd.qcut to create quantile-based bins with equal number of records in each bin
        df[col_name] = pd.qcut(df[col_name], q=num_bins, labels=False)

        # Update the bin labels to be more descriptive
        df[col_name] = df[col_name].map(lambda x: bin_labels[x])
    
        # Convert the column to object dtype
        df[col_name] = df[col_name].astype('object')

        return df
    
    def rebin_campaign(self, x):
        if x < 7:
            return str(x)
        elif x < 10:
            return '7-9'
        elif x < 15:
            return '10-14'
        else:
            return '15+'
        
    def rebin_pdays(self, x):
        if x == -1:
            return 'None'
        else:
            return str(30*round(x/30))
        
    def rebin_previous(self, x):
        if x < 5:
            return str(x)
        elif x < 7:
            return '5-6'
        elif x < 10:
            return '7-9'
        else:
            return '10+'
    
    def apply_rebinning(self, df):
        df['campaign'] = df['campaign'].apply(self.rebin_campaign)
        df['pdays'] = df['pdays'].apply(self.rebin_pdays)
        df['previous'] = df['previous'].apply(self.rebin_previous)
        return df
    
    def transform_df(self, df):
        # Drop non-necessary columns
        df = df.drop(columns=['poutcome'], axis=1)

        # Drop duplicates rows
        df = df.drop_duplicates()

        # Encoding target variable
        main_label = 'y'
        df = df[df[main_label].isin(['yes', 'no'])]
        df[main_label] = (df[main_label] == 'yes').astype(int)

        # dtype conversion
        df['day'] = df['day'].astype(str)

        # Apply bin_column to group in classes
        for col in ['age', 'balance', 'duration']:
            try:
                df = self.bin_column(df, col)
            except:
                print(f'Column {col} requires manual rebinning')

        # Apply rebinning functions
        df = self.apply_rebinning(df)

        # Codificación de etiquetas raras
        for col in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
                    'month', 'day', 'campaign', 'pdays', 'previous']:
            df[col] = df[col].fillna('None')

            encoder = RareLabelEncoder(n_categories=1, max_n_categories=70, replace_with='Other', tol=20 / df.shape[0])

            df[col] = encoder.fit_transform(df[col])

        return df

   
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            logging.info("Completed loading of train/test data")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.transform_df()

            target_columns_name = "y"

            # X and y in Train dataset
            input_feature_train_df = train_df.drop(columns=[target_columns_name], axis = 1)
            target_feature_train_df = train_df[target_columns_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]

            logging.info(f"Saved preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)


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
        
