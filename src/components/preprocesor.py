import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import pickle
from feature_engine.encoding import RareLabelEncoder


def bin_column(df, col_name, num_bins=5):

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


# Definir la función de binning manual
def rebin_campaign(x):
    if x < 7:
        return str(x)
    elif x < 10:
        return '7-9'
    elif x < 15:
        return '10-14'
    else:
        return '15+'

def rebin_pdays(x):
    if x == -1:
        return 'None'
    else:
        return str(30*round(x/30))

def rebin_previous(x):
    if x < 5:
        return str(x)
    elif x < 7:
        return '5-6'
    elif x < 10:
        return '7-9'
    else:
        return '10+'

# Definir la clase del preprocesador personalizado
class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, df):
        return self
    
    def transform(self, df):
        # Df copy
        df_copy = df.copy()

        # Drop non necesary columns
        df_copy = df_copy.drop(column=['poutcome'], axis=1)

        # Drop duplicates rows
        df_copy = df_copy.drop_duplicates()

        # Encoding target variable
        main_label = 'y'
        df_copy = df_copy[df[main_label].isin(['yes', 'no'])]
        df_copy[main_label] = (df_copy[main_label]=='yes').astype(int)
        
        # dtype conversion
        df_copy['day'] = df_copy['day'].astype(str)
        
        # apply bin_column to group in classes 
        for col in ['age', 'balance', 'duration']:
            try:
                df_copy = bin_column(df_copy, col)
            except:
                print(f'Column {col} requires manual rebinning')

        df_copy['campaign'] = df_copy['campaign'].apply(rebin_campaign)
        df_copy['pdays'] = df_copy['pdays'].apply(rebin_pdays)
        df_copy['previous'] = df_copy['previous'].apply(rebin_previous)
        
        # Codificación de etiquetas raras
        for col in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 
                    'month', 'day', 'campaign', 'pdays', 'previous']:
            
            df_copy[col] = df_copy[col].fillna('None')

            encoder = RareLabelEncoder(n_categories=1, max_n_categories=70, replace_with='Other', tol=20/df.shape[0])

            df_copy[col] = encoder.fit_transform(df_copy[col])
        
        return df_copy