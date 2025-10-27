import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import yaml
import os
import sys
from src.logger import logging
from sklearn.preprocessing import MinMaxScaler


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise



def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise


def train_model(X_train: np.array , y_train: np.array) -> RandomForestClassifier:
    """Train the logistic reg model """
    try:
        params = load_params('./params.yaml')
        model_params = params['random_forest']
        depth = model_params['max_depth']
        n_estimators = model_params['n_estimators']
        min_samples_split = model_params['min_samples_split']
        min_samples_leaf = model_params['min_samples_leaf']

        logging.info("training the model ")
        model = RandomForestClassifier(n_estimators = n_estimators ,max_depth = depth)
        model.fit(X_train , y_train)

        logging.info('Model training completed')
        return model
    except Exception as e:
        logging.error('Error during model training: %s', e)
        raise


def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info('Model saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:
        train_data = load_data("/Users/arpitgupta/Desktop/MLOPS+LLMOPS/MLOPS-LLMOPS/Datas/processed/train_preprocessed.csv")
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        model = train_model(X_train , y_train)


        # Save model and PowerTransformer
        save_model(model, 'models/model.pkl')

    except Exception as e:
        logging.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()



