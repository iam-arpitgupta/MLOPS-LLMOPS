"""
MLflow Model Registry - Register and manage model versions
"""
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import yaml


def load_params(params_path: str) -> dict:
    """Load parameters from yaml file"""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        print(f'Parameters loaded from {params_path}')
        return params
    except Exception as e:
        print(f'Error loading params: {e}')
        raise


def get_best_run(experiment_name: str, metric: str = "accuracy"):
    """Get the best run from an experiment based on a metric"""
    try:
        print(f"\nSearching for best run in experiment: {experiment_name}")
        
        client = MlflowClient()
        
        # Get experiment
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found!")
        
        experiment_id = experiment.experiment_id
        
        # Search for runs
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1
        )
        
        if not runs:
            raise ValueError(f"No runs found in experiment '{experiment_name}'")
        
        best_run = runs[0]
        
        print(f"Best Run ID: {best_run.info.run_id}")
        print(f"Best {metric}: {best_run.data.metrics[metric]:.4f}")
        print(f"Run Name: {best_run.data.tags.get('mlflow.runName', 'N/A')}")
        
        return best_run
    
    except Exception as e:
        print(f'Error getting best run: {e}')
        raise


def register_model(run_id: str, model_name: str, model_path: str = "model"):
    """Register a model to MLflow Model Registry"""
    try:
        print(f"\nRegistering model '{model_name}' from run {run_id}...")
        
        # Model URI
        model_uri = f"runs:/{run_id}/{model_path}"
        
        # Register model
        model_version = mlflow.register_model(model_uri, model_name)
        
        print(f"Model registered successfully!")
        print(f"Model Name: {model_version.name}")
        print(f"Model Version: {model_version.version}")
        
        return model_version
    
    except Exception as e:
        print(f'Error registering model: {e}')
        raise


def transition_model_stage(model_name: str, version: int, stage: str):
    """Transition model to a specific stage (Staging/Production/Archived)"""
    try:
        print(f"\nTransitioning model '{model_name}' version {version} to {stage}...")
        
        client = MlflowClient()
        
        # Transition model version stage
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=False
        )
        
        print(f"Model transitioned to {stage} successfully!")
        
    except Exception as e:
        print(f'Error transitioning model stage: {e}')
        raise


def add_model_description(model_name: str, version: int, description: str):
    """Add description to a model version"""
    try:
        client = MlflowClient()
        
        client.update_model_version(
            name=model_name,
            version=version,
            description=description
        )
        
        print(f"Description added to model version {version}")
        
    except Exception as e:
        print(f'Error adding model description: {e}')
        raise


def list_registered_models():
    """List all registered models"""
    try:
        print("\nListing all registered models...")
        
        client = MlflowClient()
        
        # Get all registered models
        registered_models = client.search_registered_models()
        
        if not registered_models:
            print("No registered models found.")
            return
        
        for rm in registered_models:
            print(f"\nModel Name: {rm.name}")
            print(f"Latest Versions:")
            for version in rm.latest_versions:
                print(f"  - Version: {version.version}, Stage: {version.current_stage}")
        
    except Exception as e:
        print(f'Error listing models: {e}')
        raise


def get_production_model(model_name: str):
    """Get the production version of a model"""
    try:
        client = MlflowClient()
        
        # Get production version
        production_versions = client.get_latest_versions(model_name, stages=["Production"])
        
        if not production_versions:
            print(f"No production version found for model '{model_name}'")
            return None
        
        prod_version = production_versions[0]
        
        print(f"\nProduction Model:")
        print(f"Model Name: {prod_version.name}")
        print(f"Version: {prod_version.version}")
        print(f"Run ID: {prod_version.run_id}")
        
        return prod_version
    
    except Exception as e:
        print(f'Error getting production model: {e}')
        raise


def main():
    """Main model registry pipeline"""
    try:
        print("=" * 50)
        print("Starting Model Registry Pipeline")
        print("=" * 50)
        
        # Load parameters
        params = load_params('params.yaml')
        
        # MLflow setup
        mlflow.set_tracking_uri(params['mlflow']['tracking_uri'])
        
        experiment_name = params['mlflow']['experiment_name']
        model_name = params['mlflow']['registered_model_name']
        
        # Get best run
        best_run = get_best_run(experiment_name, metric="accuracy")
        
        # Register model
        model_version = register_model(
            run_id=best_run.info.run_id,
            model_name=model_name
        )
        
        # Add description
        description = f"Crop recommendation model trained with accuracy: {best_run.data.metrics['accuracy']:.4f}"
        add_model_description(model_name, model_version.version, description)
        
        # Transition to Staging first
        transition_model_stage(model_name, model_version.version, "Staging")
        
        print("\n" + "=" * 50)
        print("Model registered and moved to Staging!")
        print(f"To promote to Production, run:")
        print(f"  python model_registry.py --promote {model_name} {model_version.version}")
        print("=" * 50)
        
        # List all registered models
        list_registered_models()
        
        print("\nModel Registry Pipeline Completed Successfully!")
        
    except Exception as e:
        print(f'Failed to complete model registry: {e}')
        raise


if __name__ == "__main__":
    import sys
    
    # Check if promoting to production
    if len(sys.argv) == 4 and sys.argv[1] == "--promote":
        params = load_params('params.yaml')
        mlflow.set_tracking_uri(params['mlflow']['tracking_uri'])
        
        model_name = sys.argv[2]
        version = int(sys.argv[3])
        
        transition_model_stage(model_name, version, "Production")
        print(f"\nModel {model_name} v{version} promoted to Production!")
    else:
        main()