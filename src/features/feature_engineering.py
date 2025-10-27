import numpy as np
import pandas as pd
import os
import sys
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler , LabelEncoder
from src.logger import logging
import yaml
import pickle 

def load_params(params_path: str) -> dict:
    """Load parameters from yaml file"""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        print(f'Parameters retrieved from {params_path}')
        return params
    except Exception as e:
        print(f'Error loading params: {e}')
        raise

def preprocess_data(df: pd.DataFrame, test_size: float, scalers_path: str, encoders_path: str):
    try:
        logging.info("Starting the preprocessing Step")
        X = df.drop('label' , axis = 1)
        y = df['label']

        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)

        #  Save label encoder
        os.makedirs(encoders_path, exist_ok=True)
        encoder_file = os.path.join(encoders_path, 'label_encoder.pkl')
        with open(encoder_file, 'wb') as f:
            pickle.dump(encoder, f)

        logging.info("Starting up with the train test split ")
        X_train , X_test , y_train, y_test = train_test_split(X , y_encoded , test_size = 0.2 , random_state = 42)
        logging.info("Done with the splitting ")

        # normalize the feature 
        logging.info("Starting with the scaling ")
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        logging.info("Done with the scailing  ")

        logging.info(f"X_train_scaled shape : {X_train_scaled.shape} and X_test_scaled {X_train_scaled.shape}")

        #  Save label encoder
        os.makedirs(scalers_path, exist_ok=True)
        scaler_file = os.path.join(scalers_path, 'scaler.pkl')
        with open(scaler_file, 'wb') as f:
            pickle.dump(encoder, f)

        # Convert scaled arrays back to DataFrames with original column names
        logging.info("Converting the trained array into dataframe")
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
        logging.info("Converting the test array into dataframe")
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)
        
        # add the encoder layer 
        train_data = X_train_scaled_df.copy()
        train_data['label'] = y_train
        
        test_data = X_test_scaled_df.copy()
        test_data['label'] = y_test
        
        logging.info("\nPreprocessing completed successfully!")
        return train_data, test_data
    
    except Exception as e:
        logging.info("Failed the feature engineering step")


def main():
    try:
        params = load_params('./params.yaml')
        test_size = params['data_preprocessing']['test_size']
        encoder_path = params["data_preprocessing"]["encoders_path"]
        scaler_path = params["data_preprocessing"]["scalers_path"]
        # test_size = .20
        preprocess_data_file_path = params['data_preprocessing']['processed_data_path']
        raw_data = params['data_ingestion']['raw_data_path']

        # Fetch the raw data
        df = pd.read_csv(raw_data)
        # df = pd.read_csv('./data/raw/creditcard.csv')
        logging.info('Raw data loaded successfully')
        
        logging.info("preprocessing the data ")
        # Preprocess data
        train_data, test_data = preprocess_data(
            df, test_size , encoder_path , scaler_path
        )

        logging.info("saving the preprocessing csv's ")
        # Save processed data after train-test split        
        train_data.to_csv(os.path.join(preprocess_data_file_path, "train_preprocessed.csv"), index=False)
        test_data.to_csv(os.path.join(preprocess_data_file_path, "test_preprocessed.csv"), index=False)
        logging.info('Processed train and test data saved successfully in %s', preprocess_data_file_path)

    except Exception as e:
        logging.error('Failed to complete the data preprocessing process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()


    





