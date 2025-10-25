# 🌾 Crop Advisor - MLOps Pipeline

An end-to-end MLOps pipeline for crop recommendation using classical ML models with MLflow tracking, DVC for data versioning, and Docker containerization.

## 📋 Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [MLflow Tracking](#mlflow-tracking)
- [DVC Setup](#dvc-setup)
- [Docker Deployment](#docker-deployment)
- [API Endpoints](#api-endpoints)

## ✨ Features

- **Data Versioning**: Track data changes with DVC
- **Experiment Tracking**: Log experiments with MLflow
- **Model Registry**: Version and manage models with MLflow Registry
- **Containerization**: Docker containers for training and inference
- **Reproducible Pipelines**: DVC pipeline for reproducibility
- **FastAPI Service**: RESTful API for model inference
- **Agentic AI**: Natural language explanations for predictions

## 📁 Project Structure

```
crop-advisor-mlops/
│
├── data/
│   ├── raw/                          # Raw dataset (tracked by DVC)
│   └── processed/                    # Processed data (tracked by DVC)
│
├── src/
│   ├── data/
│   │   ├── data_ingestion.py        # Data loading and validation
│   │   └── data_preprocessing.py    # Feature engineering
│   ├── models/
│   │   ├── train_model.py           # Model training with MLflow
│   │   └── model_registry.py        # MLflow model registration
│   └── api/
│       └── main.py                   # FastAPI application
│
├── artifacts/
│   ├── models/                       # Trained models (tracked by DVC)
│   ├── scalers/                      # Saved scalers
│   └── encoders/                     # Saved encoders
│
├── docker/
│   ├── Dockerfile.training           # Training container
│   ├── Dockerfile.mlflow             # MLflow server container
│   └── Dockerfile.api                # FastAPI container
│
├── mlflow/                           # MLflow artifacts
├── params.yaml                       # Configuration parameters
├── dvc.yaml                          # DVC pipeline definition
├── docker-compose.yml                # Docker orchestration
├── requirements.txt                  # Python dependencies
├── Makefile                          # Helper commands
└── README.md                         # Project documentation
```

## 🔧 Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Git
- DVC (Data Version Control)

## 📦 Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd crop-advisor-mlops
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
make install
# or
pip install -r requirements.txt
```

### 4. Setup DVC
```bash
make setup-dvc
# or
dvc init
```

## 🚀 Usage

### Option 1: Run Locally

#### Step 1: Data Ingestion
```bash
python -m src.data.data_ingestion
```

#### Step 2: Data Preprocessing
```bash
python -m src.data.data_preprocessing
```

#### Step 3: Model Training
```bash
python -m src.models.train_model
```

#### Step 4: Model Registry
```bash
python -m src.models.model_registry
```

Or run the complete pipeline:
```bash
make train
```

### Option 2: Run with DVC Pipeline
```bash
dvc repro
# or
make run-pipeline
```

### Option 3: Run with Docker
```bash
# Build containers
make docker-build

# Start all services
make docker-up

# View logs
make docker-logs

# Stop services
make docker-down
```

## 📊 MLflow Tracking

### Start MLflow UI
```bash
make mlflow-ui
# or
mlflow ui --host 0.0.0.0 --port 5000
```

Access MLflow UI at: `http://localhost:5000`

### MLflow Features
- **Experiment Tracking**: Compare model performance
- **Model Registry**: Version and stage models
- **Artifact Storage**: Store models and plots
- **Metrics Logging**: Track accuracy, precision, recall, F1

### Promote Model to Production
```bash
python -m src.models.model_registry --promote crop-advisor-model 1
```

## 💾 DVC Setup

### Initialize DVC with Remote Storage

#### For S3:
```bash
dvc remote add -d storage s3://your-bucket-name/crop-advisor
dvc remote modify storage access_key_id YOUR_ACCESS_KEY
dvc remote modify storage secret_access_key YOUR_SECRET_KEY
```

#### For Google Cloud Storage:
```bash
dvc remote add -d storage gs://your-bucket-name/crop-advisor
```

#### For Azure Blob:
```bash
dvc remote add -d storage azure://your-container/crop-advisor
```

### Track Data with DVC
```bash
# Add data to DVC tracking
dvc add data/raw/Crop_recommendation.csv
dvc add data/processed/
dvc add artifacts/models/

# Commit DVC files to Git
git add data/.gitignore data/raw/Crop_recommendation.csv.dvc
git commit -m "Track data with DVC"

# Push data to remote storage
dvc push
```

### Pull Data from Remote
```bash
dvc pull
```

## 🐳 Docker Deployment

### Services

1. **PostgreSQL**: Database for MLflow backend
2. **MLflow Server**: Experiment tracking and model registry
3. **Training Service**: Runs training pipeline
4. **API Service**: FastAPI inference endpoint

### Docker Commands

```bash
# Build all containers
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f training
docker-compose logs -f api

# Stop services
docker-compose down

# Remove volumes
docker-compose down -v
```

### Access Services
- MLflow UI: `http://localhost:5000`
- FastAPI: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`

## 🔌 API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Predict Crop
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 20.8,
    "humidity": 82.0,
    "ph": 6.5,
    "rainfall": 202.9
  }'
```

Response:
```json
{
  "crop": "rice",
  "confidence": 0.95,
  "explanation": "Based on soil pH and humidity, rice is ideal this season."
}
```

## 📈 Monitoring

### View Training Metrics
```bash
# In MLflow UI
http://localhost: