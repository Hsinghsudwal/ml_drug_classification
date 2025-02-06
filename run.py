from pipelines.training_pipeline import pipeline

from src.ml_drug_classification.model_staging import model_stage_staging

from src.ml_drug_classification.model_production import model_stage_production

from test.test import test_evaluate


def run():
    pipeline()

    model_stage_staging()


def test_prod():

    result = test_evaluate()
    if result == "Model needs Retraining":
        print("Need re-training")
        pipeline()
    else:
        print("Model push to Production")
        model_stage_production()


if __name__ == "__main__":
    run()

    test_prod()
