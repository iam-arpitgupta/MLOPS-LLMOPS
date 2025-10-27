import mlflow 
import mlflow.sklearn
import dagshub 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler , LabelEncoder ,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score

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

FE_TECH = {
    'MinMaxScaler': MinMaxScaler(),
    'StandardScaler': StandardScaler()
}

# Encoder = {
#     'LabelEncoder':  LabelEncoder(),
#     'OneHotEnconder': OneHotEncoder()
# }

ALGORITHMS = {
    'LogisticRegression': LogisticRegression(),
    'RandomForest': RandomForestClassifier(),
    "DecisionTree":DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB(),
    # "xgboost" : Xgboost()
}


def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df 
    except Exception as e:
        print("error loading {e}")
        raise


def train_and_evaluate(df):
    X = df.drop('label' , axis = 1)
    y = df['label']

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train , X_test , y_train , y_test = train_test_split(X , y_encoded , test_size = CONFIG['test_size'] , random_state=42)

    with mlflow.start_run(run_name = "Feature Engoneering and models") as parent_run:
        for fe_name , fe_method in FE_TECH.items():
            try:
                X_train_transformed = fe_method.fit_transform(X_train)
                X_test_transformed = fe_method.transform(X_test)


                # for e_name , e_method in Encoder.items():
                #     y_encoded = e_method.fit_transform(y)


                for algo , algo_model in ALGORITHMS.items():
                    with mlflow.start_run(run_name = f"{algo_model} with {fe_name} " , nested = True) as child_run:
                        try:
                            model = algo_model
                            model.fit(X_train_transformed, y_train)
                            y_pred = model.predict(X_test_transformed)

                             # Log preprocessing parameters
                            mlflow.log_params({
                                "feature engineering technique": fe_name,
                                "algorithm": algo,
                                "test_size": CONFIG["test_size"],
                                # "Encoder" : e_name,
                            })
                            metrics = {
                                "accuracy": accuracy_score(y_test, y_pred),
                                "precision": precision_score(y_test, y_pred ,average='weighted' ),
                                "recall": recall_score(y_test, y_pred,average='weighted'),
                                "f1_score": f1_score(y_test, y_pred,average='weighted'),
                            }
                            mlflow.log_metrics(metrics)
                            # log_model_params(algo_name, model)
                            mlflow.sklearn.log_model(model, "model")

                            print(f"\nFeature Engineering: {fe_name} | |  Algorithm: {algo}")
                            print(f"Metrics: {metrics}")
                        except Exception as e:
                            print(f"Error training {algo} with {fe_name}: {e}")
                            mlflow.log_param("error", str(e))
            except Exception as fe_error:
                print(f"Error applying {fe_name}: {fe_error}")
                mlflow.log_param("fe_error", str(fe_error))


# ========================== EXECUTION ==========================
if __name__ == "__main__":
    df = load_data(CONFIG["data_path"])
    train_and_evaluate(df)




            

