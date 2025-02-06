import mlflow
import mlflow
from mlflow.tracking import MlflowClient
import joblib


def load_model():
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    EXPERIMENT_NAME = "predict_drug"
    mlflow.set_experiment(EXPERIMENT_NAME)
    MODEL_NAME = "best_model"
    PROD = "Production"

    client = MlflowClient()

    # latest model version in stage
    versions = client.get_latest_versions(MODEL_NAME, stages=[PROD])

    run_id = versions[0].run_id

    # load model
    model_uri = f"runs:/{run_id}/{MODEL_NAME}"
    print(f"Model URI: {model_uri}")

    loaded_model = mlflow.pyfunc.load_model(model_uri)

    # Save model
    model_save_path = "app_model.joblib"

    joblib.dump(loaded_model, model_save_path)


if __name__ == "__main__":
    load_model()
