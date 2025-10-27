import json
import mlflow
import logging
import sys
from src.logger import logging
import dagshub
from dotenv import load_dotenv
import os 

load_dotenv()


CONFIG = {
    "data_path": "/Users/arpitgupta/Desktop/MLOPS+LLMOPS/MLOPS-LLMOPS/notebooks/Crop_recommendation.csv",
    "test_size": 0.2,
    "mlflow_tracking_uri":"https://dagshub.com/thearpitgupta2003/MLOPS-LLMOPS.mlflow",
    "dagshub_repo_owner":"thearpitgupta2003",
    "dagshub_repo_name":"MLOPS-LLMOPS",
    "experiment_name":"Naive Bayes",
}


mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
dagshub.init(repo_owner=CONFIG["dagshub_repo_owner"], repo_name=CONFIG["dagshub_repo_name"], mlflow=True)
mlflow.set_experiment(CONFIG["experiment_name"])


# Below code block is for production use
# -------------------------------------------------------------------------------------
# # Set up DagsHub credentials for MLflow tracking
# dagshub_token = os.getenv("CAPSTONE_TEST")
# if not dagshub_token:
#     raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# dagshub_url = "https://dagshub.com"
# repo_owner = "thearpitgupta2003"
# repo_name = "MLOPS-LLMOPS"

# # Set up MLflow tracking URI
# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise


def registar_all_models(model_name : str , model_info : dict , scaler_name: str , encoder_name : str , scalers_path : str,encoders_path : str):
    """Registar the model , scaler and the encoder """
    try:
        client = mlflow.tracking.MlflowClient()

        # registar the model 
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        logging.info(model_uri)

        model_version = mlflow.register_model(model_uri, model_name)

        client.transition_model_version_stage(
            name = model_name ,
            version = model_version.version,
            stage = "Staging"
        )
        logging.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
        
        # Log the scaler as an artifact
        mlflow.log_artifact(scalers_path, artifact_path="preprocessing")

        # Register PowerTransformer in MLflow Model Registry
        scalers_uri = f"runs:/{model_info['run_id']}/preprocessing/{os.path.basename(scalers_path)}"
        scalers_version = mlflow.register_model(scalers_uri, scalers_path)

        client.transition_model_version_stage(
            name = scaler_name ,
            version = scalers_version.version,
            stage = "Staging"
        )


        # Register PowerTransformer in MLflow Model Registry
        encoders_uri = f"runs:/{model_info['run_id']}/preprocessing/{os.path.basename(encoders_path)}"
        encoders_version = mlflow.register_model(encoders_uri, encoders_path)



        client.transition_model_version_stage(
            name = scaler_name ,
            version = scalers_version.version,
            stage = "Staging"
        )

    except Exception as e:
        logging.error('Error during model and transformer registration: %s', e)
        raise



def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)

        model_name = "my_model"
        scalers_name = "MinMaxScaler"
        scalers_path = "models/scaler.pkl" 
        encoders_name = "LabelEncoder"
        encoders_path = "models/label_encoder.pkl"
        
        # Ensure this file exists

        registar_all_models(model_name, model_info, scalers_name, encoders_name,scalers_path,encoders_path )
    
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()