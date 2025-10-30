import unittest
import mlflow
import dagshub
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

CONFIG = {
    "mlflow_tracking_uri": "https://dagshub.com/thearpitgupta2003/MLOPS-LLMOPS.mlflow",
    "dagshub_repo_owner": "thearpitgupta2003",
    "dagshub_repo_name": "MLOPS-LLMOPS",
    "experiment_name": "Naive Bayes",
}


class TestCropModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):  #  FIXED spelling: setUpClass (not setupClass)
        """Set up test environment once before running tests."""
        log.info("Initializing MLflow + DagsHub setup...")

        # Initialize MLflow + DagsHub
        mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
        dagshub.init(
            repo_owner=CONFIG["dagshub_repo_owner"],
            repo_name=CONFIG["dagshub_repo_name"],
            mlflow=True
        )
        mlflow.set_experiment(CONFIG["experiment_name"])

        cls.model_name = "my_model"
        cls.model_version = cls.get_latest_model_version(cls.model_name)

        if cls.model_version is None:
            raise RuntimeError(
                f"No model found in 'Staging' for {cls.model_name}. "
                f"Promote one in MLflow before testing."
            )

        cls.model_uri = f"models:/{cls.model_name}/{cls.model_version}"
        cls.model = mlflow.pyfunc.load_model(cls.model_uri)

        # ✅ Load artifacts
        cls.scaler = pickle.load(open("models/scaler.pkl", "rb"))
        cls.encoder = pickle.load(open("models/encoder.pkl", "rb"))
        cls.test_data = pd.read_csv("Datas/processed/test_preprocessed.csv")

        log.info("Setup completed successfully.")

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        """Fetch the latest model version in a given MLflow stage."""
        client = mlflow.MlflowClient()
        try:
            versions = client.get_latest_versions(model_name, stages=[stage])
            if versions:
                return versions[0].version
        except Exception as e:
            log.warning(f"Could not fetch model version: {e}")
        return None

    def test_model_loaded_properly(self):
        """Ensure model loads correctly."""
        self.assertIsNotNone(self.model, "Model should not be None after loading.")

    def test_model_signature(self):
        """Check model input-output signature."""
        sample_input = self.test_data.iloc[:1, :-1]
        transformed_input = self.scaler.transform(sample_input)
        input_df = pd.DataFrame(transformed_input, columns=sample_input.columns)

        prediction = self.model.predict(input_df)

        self.assertEqual(input_df.shape[1], self.scaler.n_features_in_)
        self.assertEqual(len(prediction), 1)

    def test_model_performance(self):
        """Evaluate model accuracy, precision, recall, and F1."""
        X_test = self.test_data.iloc[:, :-1]
        y_test = self.test_data.iloc[:, -1]

        X_test_transformed = self.scaler.transform(X_test)
        X_test_df = pd.DataFrame(X_test_transformed, columns=X_test.columns)

        y_pred = self.model.predict(X_test_df)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        log.info(
            f"Model Performance - Accuracy: {accuracy:.4f}, "
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
        )

        for metric, value in {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }.items():
            self.assertGreaterEqual(value, 0.5, f"{metric} below threshold (0.5)")


if __name__ == "__main__":
    unittest.main()
