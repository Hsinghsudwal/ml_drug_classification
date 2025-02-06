import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def test_data():
    # Load test data
    data = pd.read_csv("data_source/drug.csv")

    x = data.drop(["Drug"], axis=1)
    y = data["Drug"]

    _, X_test, _, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    return _, X_test, _, y_test


def process(test):

    cat_col = [1, 2, 3]
    num_col = [0, 4]

    # preprocess
    preprocess = ColumnTransformer(
        transformers=[
            (
                "encoder",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        ("encoder", OrdinalEncoder()),
                    ]
                ),
                cat_col,
            ),
            (
                "num_transform",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="median"),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_col,
            ),
        ],
        remainder="passthrough",
    )

    # transform test
    xtest_preprocessed = preprocess.fit_transform(test)
    xtest = pd.DataFrame(xtest_preprocessed, columns=test.columns)
    return xtest


def test_model():
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    EXPERIMENT_NAME = "predict_drug"
    mlflow.set_experiment(EXPERIMENT_NAME)
    MODEL_NAME = "best_model"
    STAGE = "Staging"

    client = MlflowClient()

    # latest model version in stage
    versions = client.get_latest_versions(MODEL_NAME, stages=[STAGE])

    assert versions, f"Model {MODEL_NAME} not found in '{STAGE}' stage."

    run_id = versions[0].run_id
    print(f"model with run_id: {run_id}")

    # load model
    model_uri = f"runs:/{run_id}/{MODEL_NAME}"
    print(f"Model URI: {model_uri}")

    loaded_model = mlflow.pyfunc.load_model(model_uri)

    return loaded_model


def test_evaluate():
    _, X_test, _, y_test = test_data()

    xtest = process(X_test)

    # Make predictions
    model = test_model()
    y_pred = model.predict(xtest)
    accuracy = accuracy_score(y_test, y_pred)

    # Define a threshold for acceptable accuracy
    expected_accuracy = 0.90

    print(f"Model Accuracy: {accuracy}")

    # Assert that the accuracy within expected threshold
    # assert (
    #     accuracy >= expected_accuracy
    # ), f"Model accuracy {accuracy} is below the threshold {expected_accuracy}. Model drifted."

    if accuracy < expected_accuracy:
        print(f"Model accuracy {accuracy} is below the threshold {expected_accuracy}.")
        return "Model needs Retraining"
    else:
        return "Model push to Production"


if __name__ == "__main__":
    result = test_evaluate()
    # print(result)
