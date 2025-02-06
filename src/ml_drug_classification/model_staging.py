from src.constants import *
from mlflow.tracking import MlflowClient
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment(EXPERIMENT_NAME)


def model_stage_staging():
    try:

        client = MlflowClient()
        latest_versions = client.get_latest_versions(MODEL_NAME, stages=["None"])

        for version in latest_versions:
            model_version = version.version

        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=model_version,
            stage=STAGE,
            archive_existing_versions=True,
        )

        print("Stage Staging Complete")
    except Exception as e:
        raise Exception(e)


# if __name__ == "__main__":
#     model_stage_staging()
