from mlflow.tracking import MlflowClient
import mlflow
from src.constants import *


mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment(EXPERIMENT_NAME)


def model_stage_production():

    try:

        client = MlflowClient()

        # Get the latest model stage-staging
        stage_versions = client.get_latest_versions(MODEL_NAME, stages=[STAGE])

        latest_staging_version = stage_versions[0]
        staging_version_number = latest_staging_version.version

        if not stage_versions:
            return "Staging-stage doesn't exist"

        new_prod_version = client.get_latest_versions(MODEL_NAME, stages=[PROD])

        if new_prod_version:
            current_production_version = new_prod_version[0]
            production_version_number = current_production_version.version

            # Transition the current Production model to Archived
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=production_version_number,
                stage=ARCHIVED,
                archive_existing_versions=False,
            )
            print("Previous Production Model Is Archived")

        # Transition the latest Staging model to Production
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=staging_version_number,
            stage=PROD,
            archive_existing_versions=False,
        )

        print("Stage New Production Model Complete")

    except Exception as e:
        raise Exception(e)


# if __name__ == "__main__":
#     model_stage_production()
