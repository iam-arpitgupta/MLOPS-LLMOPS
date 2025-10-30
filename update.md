# 🌾 Crop Advisor MLOps - Complete Project Architecture

## 📊 Project Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          CROP ADVISOR MLOPS PIPELINE                             │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              STAGE 1: DATA LAYER                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────┐
    │  Kaggle Dataset  │
    │  Crop_recommendation.csv │
    └─────────┬────────┘
              │
              ▼
    ┌──────────────────────────┐
    │  data_ingestion.py       │◄────── params.yaml
    │  - Load CSV              │
    │  - Validate Schema       │
    │  - Check Nulls/Outliers  │
    └─────────┬────────────────┘
              │
              ▼
    ┌──────────────────────────┐
    │  data/raw/               │
    │  Crop_recommendation.csv │ ◄──────── DVC Tracked
    └─────────┬────────────────┘
              │
              ▼
    ┌──────────────────────────┐
    │  feature_engineering.py  │
    │  - NPK Ratios            │
    │  - Environmental Features│
    │  - Categorical Features  │
    └─────────┬────────────────┘
              │
              ▼
    ┌──────────────────────────┐
    │  data_preprocessing.py   │◄────── params.yaml
    │  - Label Encoding        │
    │  - MinMax Scaling        │
    │  - Train/Test Split      │
    └─────────┬────────────────┘
              │
              ├─────────────────────────────┐
              │                             │
              ▼                             ▼
    ┌──────────────────┐          ┌──────────────────┐
    │ train_data.csv   │          │ test_data.csv    │
    └──────────────────┘          └──────────────────┘
              │                             │
              │    ┌────────────────┐       │
              └───►│ artifacts/     │◄──────┘
                   │ - scalers/     │
                   │ - encoders/    │ ◄──────── DVC Tracked
                   └────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────┐
│                          STAGE 2: MODEL TRAINING LAYER                           │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────┐
    │  train_data.csv          │
    │  test_data.csv           │
    └─────────┬────────────────┘
              │
              ▼
    ┌──────────────────────────────────────┐
    │  model_training.py                   │◄────── params.yaml
    │                                      │
    │  ┌────────────────────────────────┐ │
    │  │  Decision Tree Classifier      │ │
    │  │  - Criterion: entropy          │ │
    │  │  - Max Depth: 5                │ │
    │  │  - Accuracy: ~89%              │ │
    │  └────────────────────────────────┘ │
    │                                      │
    │  ┌────────────────────────────────┐ │
    │  │  Random Forest Classifier      │ │
    │  │  - N Estimators: 100           │ │
    │  │  - Max Depth: 10               │ │
    │  │  - Accuracy: ~93% ✓ BEST       │ │
    │  └────────────────────────────────┘ │
    │                                      │
    │  ┌────────────────────────────────┐ │
    │  │  XGBoost Classifier            │ │
    │  │  - N Estimators: 100           │ │
    │  │  - Learning Rate: 0.1          │ │
    │  │  - Accuracy: ~91%              │ │
    │  └────────────────────────────────┘ │
    └─────────┬────────────────────────────┘
              │
              ├──────────────────────────────┐
              │                              │
              ▼                              ▼
    ┌──────────────────┐          ┌──────────────────────┐
    │  MLflow Tracking │          │  artifacts/models/   │
    │  Experiment:     │          │  - DecisionTree.pkl  │
    │  crop-recommendation │      │  - RandomForest.pkl  │◄── DVC Tracked
    │                  │          │  - XGBoost.pkl       │
    │  Logs:           │          │  - BestModel.pkl     │
    │  - Parameters    │          │  - metrics.json      │
    │  - Metrics       │          └──────────────────────┘
    │  - Artifacts     │
    │  - Models        │
    └──────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────┐
│                        STAGE 3: MODEL REGISTRY LAYER                             │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────┐
    │  model_registry.py       │
    │                          │
    │  - Get Best Run          │
    │  - Register Model        │
    │  - Add Description       │
    │  - Manage Versions       │
    └─────────┬────────────────┘
              │
              ▼
    ┌─────────────────────────────────────────┐
    │  MLflow Model Registry                  │
    │                                         │
    │  Model: crop-advisor-model              │
    │                                         │
    │  ┌───────────────────────────────────┐ │
    │  │  Version 1: None                  │ │
    │  │  Accuracy: 89%                    │ │
    │  └───────────────────────────────────┘ │
    │                                         │
    │  ┌───────────────────────────────────┐ │
    │  │  Version 2: Staging               │ │
    │  │  Accuracy: 93%                    │ │
    │  │  Status: Testing                  │ │
    │  └───────────────────────────────────┘ │
    │                                         │
    │  ┌───────────────────────────────────┐ │
    │  │  Version 3: Production ✓          │ │
    │  │  Accuracy: 93%                    │ │
    │  │  Status: Active                   │ │
    │  └───────────────────────────────────┘ │
    └─────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────┐
│                        STAGE 4: DEPLOYMENT LAYER                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────────┐
│                    LOCAL DEVELOPMENT - DOCKER COMPOSE                           │
└────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────┐
    │                    docker-compose.yml                            │
    └─────────────────────────────────────────────────────────────────┘
              │
              ├──────────────┬──────────────┬──────────────┐
              │              │              │              │
              ▼              ▼              ▼              ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │ PostgreSQL   │  │ MLflow Server│  │ Training     │  │ FastAPI      │
    │ Container    │  │ Container    │  │ Container    │  │ Container    │
    │              │  │              │  │              │  │              │
    │ Port: 5432   │  │ Port: 5000   │  │ Runs:        │  │ Port: 8000   │
    │              │  │              │  │ - Ingestion  │  │              │
    │ Stores:      │  │ Features:    │  │ - Preprocess │  │ Endpoints:   │
    │ - MLflow     │  │ - Tracking   │  │ - Training   │  │ - /predict   │
    │   Metadata   │  │   UI         │  │ - Registry   │  │ - /health    │
    │              │  │ - Model      │  │              │  │ - /explain   │
    │              │  │   Registry   │  │              │  │              │
    └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
           │                 │                  │                  │
           └─────────────────┴──────────────────┴──────────────────┘
                                     │
                          crop-advisor-network


┌────────────────────────────────────────────────────────────────────────────────┐
│              PRODUCTION DEPLOYMENT - AWS EKS + ECR                              │
└────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────┐
    │                    Build & Push to ECR                           │
    └─────────────────────────────────────────────────────────────────┘
              │
              ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │         AWS Elastic Container Registry (ECR)                     │
    │                                                                  │
    │  ┌────────────────────────────────────────────────────────┐    │
    │  │  crop-advisor-mlflow:latest                            │    │
    │  │  crop-advisor-training:latest                          │    │
    │  │  crop-advisor-api:latest                               │    │
    │  │  crop-advisor-streamlit:latest                         │    │
    │  └────────────────────────────────────────────────────────┘    │
    └──────────────────────────────────────────────────────────────────┘
              │
              ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │      AWS Elastic Kubernetes Service (EKS) Cluster                │
    │                                                                  │
    │  ┌────────────────────────────────────────────────────────┐    │
    │  │              Master Nodes (Managed by AWS)             │    │
    │  └────────────────────────────────────────────────────────┘    │
    │                                                                  │
    │  ┌────────────────────────────────────────────────────────┐    │
    │  │                   Worker Nodes (EC2)                   │    │
    │  │                                                        │    │
    │  │  ┌──────────────────────────────────────────────┐    │    │
    │  │  │            Namespace: crop-advisor            │    │    │
    │  │  │                                               │    │    │
    │  │  │  ┌─────────────────────────────────────┐    │    │    │
    │  │  │  │  PostgreSQL Deployment              │    │    │    │
    │  │  │  │  - Replicas: 1                      │    │    │    │
    │  │  │  │  - PVC: 20Gi (EBS Volume)          │    │    │    │
    │  │  │  │  - Service: ClusterIP              │    │    │    │
    │  │  │  │  - Port: 5432                       │    │    │    │
    │  │  │  └─────────────────────────────────────┘    │    │    │
    │  │  │                                               │    │    │
    │  │  │  ┌─────────────────────────────────────┐    │    │    │
    │  │  │  │  MLflow Server Deployment           │    │    │    │
    │  │  │  │  - Replicas: 1                      │    │    │    │
    │  │  │  │  - Image: ECR/mlflow:latest         │    │    │    │
    │  │  │  │  - Service: LoadBalancer            │    │    │    │
    │  │  │  │  - Port: 5000                       │    │    │    │
    │  │  │  │  - Storage: S3 (Artifacts)          │    │    │    │
    │  │  │  └─────────────────────────────────────┘    │    │    │
    │  │  │                                               │    │    │
    │  │  │  ┌─────────────────────────────────────┐    │    │    │
    │  │  │  │  FastAPI Deployment                 │    │    │    │
    │  │  │  │  - Replicas: 3 (Auto-scaling)       │    │    │    │
    │  │  │  │  - Image: ECR/api:latest            │    │    │    │
    │  │  │  │  - Service: LoadBalancer            │    │    │    │
    │  │  │  │  - Port: 8000                       │    │    │    │
    │  │  │  │  - HPA: Min 3, Max 10               │    │    │    │
    │  │  │  └─────────────────────────────────────┘    │    │    │
    │  │  │                                               │    │    │
    │  │  │  ┌─────────────────────────────────────┐    │    │    │
    │  │  │  │  Streamlit Deployment               │    │    │    │
    │  │  │  │  - Replicas: 2                      │    │    │    │
    │  │  │  │  - Image: ECR/streamlit:latest      │    │    │    │
    │  │  │  │  - Service: LoadBalancer            │    │    │    │
    │  │  │  │  - Port: 8501                       │    │    │    │
    │  │  │  └─────────────────────────────────────┘    │    │    │
    │  │  │                                               │    │    │
    │  │  │  ┌─────────────────────────────────────┐    │    │    │
    │  │  │  │  Training Job (CronJob)             │    │    │    │
    │  │  │  │  - Schedule: Weekly/Monthly         │    │    │    │
    │  │  │  │  - Image: ECR/training:latest       │    │    │    │
    │  │  │  │  - Triggers model retraining        │    │    │    │
    │  │  │  └─────────────────────────────────────┘    │    │    │
    │  │  └───────────────────────────────────────────┘    │    │
    │  └────────────────────────────────────────────────────┘    │
    │                                                                  │
    │  ┌────────────────────────────────────────────────────────┐    │
    │  │              Ingress Controller (ALB)                  │    │
    │  │  - Routes traffic to services                          │    │
    │  │  - SSL/TLS Termination                                 │    │
    │  │  - Domain: crop-advisor.yourdomain.com                 │    │
    │  └────────────────────────────────────────────────────────┘    │
    └──────────────────────────────────────────────────────────────────┘
              │
              ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │                   AWS Services Integration                       │
    │                                                                  │
    │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐      │
    │  │  S3 Bucket    │  │  EBS Volumes  │  │  CloudWatch   │      │
    │  │  - Model      │  │  - PostgreSQL │  │  - Logs       │      │
    │  │  - Artifacts  │  │  - Data       │  │  - Metrics    │      │
    │  │  - DVC Remote │  │  - PVCs       │  │  - Alarms     │      │
    │  └───────────────┘  └───────────────┘  └───────────────┘      │
    │                                                                  │
    │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐      │
    │  │  IAM Roles    │  │  VPC          │  │  Route 53     │      │
    │  │  - EKS Access │  │  - Subnets    │  │  - DNS        │      │
    │  │  - ECR Pull   │  │  - Security   │  │  - Domain     │      │
    │  │  - S3 Access  │  │    Groups     │  │               │      │
    │  └───────────────┘  └───────────────┘  └───────────────┘      │
    └──────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────┐
│                      STAGE 5: USER INTERFACE LAYER                               │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────┐
    │                    app.py (Streamlit)                         │
    │                    Port: 8501                                 │
    └──────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
    ┌────────────────────────────┐  ┌────────────────────────────┐
    │  📊 Crop Prediction Tab    │  │  💬 AI Agent Chat Tab      │
    │                            │  │                            │
    │  Input Form:               │  │  Chat Features:            │
    │  - Nitrogen (N)            │  │  - "Why not wheat?"        │
    │  - Phosphorus (P)          │  │  - Fertilizer advice       │
    │  - Potassium (K)           │  │  - Irrigation planning     │
    │  - Temperature             │  │  - Feature reasoning       │
    │  - Humidity                │  │                            │
    │  - pH                      │  │  Agent Capabilities:       │
    │  - Rainfall                │  │  - Natural language        │
    │                            │  │  - Contextual responses    │
    │  Output:                   │  │  - Farming recommendations │
    │  - Recommended Crop        │  │                            │
    │  - Confidence Score        │  │                            │
    │  - Natural Report          │  │                            │
    │  - Fertilizer Advice       │  │                            │
    │  - Irrigation Plan         │  │                            │
    └────────────────────────────┘  └────────────────────────────┘
                    │                           │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │  Load Model & Artifacts: │
                    │  - RandomForest.pkl      │
                    │  - minmax_scaler.pkl     │
                    │  - label_encoder.pkl     │
                    └──────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────┐
│                         STAGE 6: MLOPS TOOLS LAYER                               │
└─────────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────┐  ┌────────────────────────────┐
│         DVC                │  │         MLflow             │
│  Data Version Control      │  │  Experiment Tracking       │
│                            │  │                            │
│  Tracks:                   │  │  Tracks:                   │
│  - data/raw/*.csv          │  │  - Hyperparameters         │
│  - data/processed/         │  │  - Metrics (accuracy, F1)  │
│  - artifacts/models/       │  │  - Artifacts (models)      │
│  - artifacts/scalers/      │  │  - Feature importance      │
│  - artifacts/encoders/     │  │                            │
│                            │  │  Registry:                 │
│  Remote Storage:           │  │  - Model versions          │
│  - S3 / GCS / Azure        │  │  - Stage transitions       │
│  - Local / Network         │  │  - Model metadata          │
│                            │  │                            │
│  Commands:                 │  │  UI: localhost:5000        │
│  - dvc add                 │  │                            │
│  - dvc push                │  │                            │
│  - dvc pull                │  │                            │
│  - dvc repro               │  │                            │
└────────────────────────────┘  └────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────┐
│                      STAGE 7: CI/CD PIPELINE (Optional)                          │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────┐
    │              GitHub Repository                            │
    └──────────────────────────────────────────────────────────┘
                            │
                            ▼
    ┌──────────────────────────────────────────────────────────┐
    │           GitHub Actions / GitLab CI                      │
    │                                                           │
    │  On Push to main:                                        │
    │  1. Run Tests                                            │
    │  2. Lint Code                                            │
    │  3. Build Docker Images                                  │
    │  4. Push to Registry                                     │
    │                                                           │
    │  On Model Update:                                        │
    │  1. Pull Latest Data (dvc pull)                         │
    │  2. Run Training Pipeline                                │
    │  3. Register Model to MLflow                             │
    │  4. Deploy to Staging                                    │
    │  5. Run Tests                                            │
    │  6. Deploy to Production (manual approval)               │
    └──────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────┐
│                      STAGE 8: MONITORING LAYER (Future)                          │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
    │  Prometheus     │  │  Grafana        │  │  Custom Logging │
    │                 │  │                 │  │                 │
    │  Metrics:       │  │  Dashboards:    │  │  Logs:          │
    │  - API Latency  │  │  - Request Rate │  │  - Predictions  │
    │  - Error Rate   │  │  - Model Perf   │  │  - User Inputs  │
    │  - Throughput   │  │  - Data Drift   │  │  - Errors       │
    └─────────────────┘  └─────────────────┘  └─────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────┐
│                           PROJECT FILE STRUCTURE                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

crop-advisor-mlops/
│
├── data/
│   ├── raw/
│   │   └── Crop_recommendation.csv          [DVC Tracked]
│   ├── processed/
│   │   ├── train_data.csv                   [DVC Tracked]
│   │   ├── test_data.csv                    [DVC Tracked]
│   │   ├── X_train.npy
│   │   └── y_train.npy
│   └── features/
│       └── data_with_features.csv
│
├── src/
│   ├── data/
│   │   ├── data_ingestion.py                [Stage 1]
│   │   ├── feature_engineering.py           [Stage 1]
│   │   └── data_preprocessing.py            [Stage 1]
│   │
│   ├── models/
│   │   ├── model_training.py                [Stage 2]
│   │   └── model_registry.py                [Stage 3]
│   │
│   └── api/
│       └── main.py                          [FastAPI Service]
│
├── artifacts/
│   ├── models/
│   │   ├── DecisionTree.pkl                 [DVC Tracked]
│   │   ├── RandomForest.pkl                 [DVC Tracked]
│   │   ├── XGBoost.pkl                      [DVC Tracked]
│   │   └── BestModel.pkl                    [DVC Tracked]
│   ├── scalers/
│   │   └── minmax_scaler.pkl                [DVC Tracked]
│   └── encoders/
│       └── label_encoder.pkl                [DVC Tracked]
│
├── docker/
│   ├── Dockerfile.training
│   ├── Dockerfile.mlflow
│   └── Dockerfile.api
│
├── mlflow/
│   ├── mlruns/                              [Experiments]
│   └── models/                              [Registry]
│
├── app.py                                   [Streamlit UI]
├── params.yaml                              [Configuration]
├── dvc.yaml                                 [DVC Pipeline]
├── docker-compose.yml                       [Orchestration]
├── requirements.txt                         [Dependencies]
├── requirements_streamlit.txt               [UI Dependencies]
├── Makefile                                 [Helper Commands]
└── README.md                                [Documentation]


┌─────────────────────────────────────────────────────────────────────────────────┐
│                           EXECUTION FLOW                                         │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│ Local Dev Mode  │
└─────────────────┘
    1. make install
    2. python -m src.data.data_ingestion
    3. python -m src.data.feature_engineering
    4. python -m src.data.data_preprocessing
    5. python -m src.models.model_training
    6. python -m src.models.model_registry
    7. streamlit run app.py

┌─────────────────┐
│ DVC Pipeline    │
└─────────────────┘
    1. dvc init
    2. dvc add data/
    3. dvc repro
    4. dvc push

┌─────────────────┐
│ Docker Mode     │
└─────────────────┘
    1. docker-compose build
    2. docker-compose up -d
    3. Access MLflow: http://localhost:5000
    4. Access API: http://localhost:8000
    5. Access Streamlit: http://localhost:8501


┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW SUMMARY                                      │
└─────────────────────────────────────────────────────────────────────────────────┘

Raw Data → Ingestion → Feature Engineering → Preprocessing → 
Split (Train/Test) → Scaling → Encoding → 
Model Training (DT, RF, XGB) → MLflow Tracking → 
Model Registry → Best Model Selection → 
Deployment (Docker/FastAPI) → 
User Interface (Streamlit) → Prediction + AI Agent


┌─────────────────────────────────────────────────────────────────────────────────┐
│                        KEY TECHNOLOGIES USED                                     │
└─────────────────────────────────────────────────────────────────────────────────┘

├── ML/Data: scikit-learn, XGBoost, pandas, numpy
├── MLOps: MLflow, DVC
├── API: FastAPI, uvicorn
├── UI: Streamlit
├── Containerization: Docker, Docker Compose
├── Database: PostgreSQL (for MLflow)
├── Monitoring: Prometheus, Grafana (optional)
├── CI/CD: GitHub Actions (optional)
└── Storage: Local/S3/GCS/Azure (for DVC)
```

---

## 🎯 Quick Start Commands

```bash
# Setup
make install
make setup-dvc

# Run Pipeline
make train

# Start Services
make docker-up

# Run Streamlit
streamlit run app.py
```

---

**📝 Note:** This diagram represents the complete end-to-end MLOps pipeline for the Crop Advisor project. Each component is modular and can be executed independently or as part of the complete workflow.2