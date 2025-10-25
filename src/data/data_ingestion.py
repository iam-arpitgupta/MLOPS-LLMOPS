"""
Data Ingestion for Crop Recommendation Dataset
"""
import pandas as pd
import os
import yaml
from pathlib import Path

def load_params(params_path: str) -> dict:
    """Load parameters from the yaml file"""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        print(f'Parameters retrieved from {params_path}')
        return params
    except FileNotFoundError:
        print(f'File not found: {params_path}')
        raise
    except yaml.YAMLError as e:
        print(f"YAML ERROR: {e}")
        raise
    except Exception as e:
        print(f'Unexpected Error: {e}')
        raise


def load_data(data_url: str) -> pd.DataFrame:
    """Load the data from the csv file"""
    try:
        df = pd.read_csv(data_url)
        print(f"Loaded CSV data from {data_url}")
        print(f"Data shape: {df.shape}")
        return df
    except pd.errors.ParserError as e:
        print(f'Parse error: {e}')
        raise
    except Exception as e:
        print(f'Could not load the file: {e}')
        raise


def validate_data(df: pd.DataFrame) -> bool:
    """Validate the data schema and quality"""
    try:
        print("Validating data...")
        
        # Expected columns
        expected_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
        
        # Check if all expected columns are present
        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            print(f"Missing columns: {missing_columns}")
            return False
        
        # Check for null values
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            print(f"Found null values:\n{null_counts[null_counts > 0]}")
            return False
        
        # Check data types
        numeric_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"Column {col} is not numeric")
                return False
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"Found {duplicates} duplicate rows")
        
        print("Data validation passed!")
        return True
    
    except Exception as e:
        print(f'Validation error: {e}')
        return False


def save_raw_data(df: pd.DataFrame, data_path: str) -> None:
    """Save raw data to the specified path"""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        
        output_file = os.path.join(raw_data_path, 'Crop_recommendation.csv')
        df.to_csv(output_file, index=False)
        print(f"Raw data saved to {output_file}")
        
    except Exception as e:
        print(f"Error while saving data: {e}")
        raise


def main():
    """Main function to run data ingestion pipeline"""
    try:
        # Load parameters
        params = load_params('params.yaml')
        data_url = params['data_ingestion']['data_url']
        data_path = params['data_ingestion']['data_path']
        
        print("=" * 50)
        print("Starting Data Ingestion Pipeline")
        print("=" * 50)
        
        # Load data
        df = load_data(data_url)
        
        # Validate data
        is_valid = validate_data(df)
        if not is_valid:
            raise ValueError("Data validation failed!")
        
        # Save raw data
        save_raw_data(df, data_path)
        
        # Print summary
        print("\n" + "=" * 50)
        print("Data Ingestion Summary:")
        print(f"Total rows: {len(df)}")
        print(f"Total columns: {len(df.columns)}")
        print(f"Crops: {df['label'].nunique()}")
        print(f"Crop types: {df['label'].unique()}")
        print("=" * 50)
        print("Data Ingestion Completed Successfully!")
        
    except Exception as e:
        print(f'Failed to complete data ingestion: {e}')
        raise


if __name__ == '__main__':
    main()