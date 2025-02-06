# BASE:
PROJECT_NAME = "Drug Classification"
AUTHOR = "Harinder Singh Sudwal"

OUTPUT = "data"

# Data Ingestion
TEST_SIZE = 0.2
RAW_FOLDER = "raw"
TRAIN_DATA_FILE = "train_data.csv"
TEST_DATA_FILE = "test_data.csv"

# Data Validation
THRESHOLD = 0.05
VALIDATE_FOLDER = "validate"
REPORT_JSON = "report.json"

# Data Transformer
TRANSFORMATION_FOLDER = "transformation"
X_TRAIN_PROCESS_DATA = "x_train_process.csv"
Y_TRAIN_PROCESS_DATA = "y_train_process.csv"
X_TEST_PROCESS_DATA = "x_test_process.csv"
Y_TEST_PROCESS_DATA = "y_test_process.csv"

# Model Trainer
MODEL_FOLDER = "model"
BEST_MODEL_NAME = "best_model.joblib"


# Model Evaluation
EVALUATION_FOLDER = "evaluate"
METRIC_JSON = "model_eval_metrics.json"
CM_MATRIX = "conf_matrix.png"
CLASS_REPORT = "classification_report.txt"

# Mlflow
SERVER_URI = "http://mlflow-server.com:5000"
EXPERIMENT_NAME = "predict_drug"
MODEL_NAME = "best_model"
ARCHIVED = "Archived"
STAGE = "Staging"
PROD = "Production"
