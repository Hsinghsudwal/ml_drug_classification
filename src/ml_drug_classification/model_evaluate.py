import pandas as pd
from src.utility import *
from src.constants import *
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    recall_score,
    f1_score,
    accuracy_score,
    precision_score,
    classification_report,
)
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import seaborn as sns
import json
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature


class ModelEvaluation:

    def __init__(self) -> None:
        pass

    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    # mlflow.set_tracking_uri(SERVER_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    def metric_score(y_test, y_pred):
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")

        return accuracy, precision, recall, f1

    def model_evaluate(self, best_model, params, X_test, y_test):
        try:
            # mlflow

            with mlflow.start_run(nested=True):
                mlflow.set_tag("Project", PROJECT_NAME)
                mlflow.set_tag("Dev", AUTHOR)

                # Predict and evaluate using the best model
                y_pred = best_model.predict(X_test)

                # print(params)
                # log params
                for param, value in params.items():
                    mlflow.log_param(param, value)

                accuracy, precision, recall, f1 = ModelEvaluation.metric_score(
                    y_test, y_pred
                )

                metrics_dict = {
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1-Score": f1,
                }

                # print("Accuracy: ", str(round(accuracy, 2) * 100) + "%")
                # print("Precision: ", round(precision, 2))
                # print("Recall: ", round(recall, 2))
                # print("F1: ", round(f1, 2))

                # log metrics
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)

                evaluate_path = os.path.join(OUTPUT, EVALUATION_FOLDER)
                os.makedirs(evaluate_path, exist_ok=True)

                # Save results to JSON file
                evaluate_filename = os.path.join(evaluate_path, METRIC_JSON)
                with open(evaluate_filename, "w") as f:

                    json.dump(
                        {
                            "metrics": metrics_dict,
                        },
                        f,
                        indent=4,
                    )
                # log metric_dic
                mlflow.log_artifact(evaluate_filename)

                # Confusion Matrix
                conf_matrix = confusion_matrix(y_test, y_pred)
                # print("Confusion Matrix:")
                # print(conf_matrix)
                sns.heatmap(conf_matrix, annot=True, fmt="g")
                plt.xlabel("Predicted Labels")
                plt.ylabel("True Labels")
                # plt.show()
                disp = ConfusionMatrixDisplay(
                    confusion_matrix=conf_matrix, display_labels=best_model.classes_
                )
                disp.plot()
                cm_path = os.path.join(OUTPUT, EVALUATION_FOLDER, str(CM_MATRIX))
                plt.savefig(cm_path, dpi=120)
                plt.close()

                # log conf_metrix
                mlflow.log_artifact(cm_path)

                # Classification Report
                class_report = classification_report(y_test, y_pred)
                # print("Classification Report:")
                # print(class_report)
                # Save as text file
                class_report_txt_path = os.path.join(
                    OUTPUT, EVALUATION_FOLDER, str(CLASS_REPORT)
                )
                with open(class_report_txt_path, "w") as f:
                    f.write(class_report)

                # log classification_report
                mlflow.log_artifact(class_report_txt_path)

                mlflow.log_artifact(__file__)

                # Log the model to MLflow
                signature = infer_signature(X_test, best_model.predict(X_test))

                mlflow.sklearn.log_model(
                    best_model, artifact_path=MODEL_NAME, signature=signature
                )

                # Register the model if needed
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
                result = mlflow.register_model(model_uri, MODEL_NAME)
                print(
                    f"Model registered with name {MODEL_NAME} and version {result.version}"
                )

                print("Model Evaluation Completed")
                return model_uri, result

        except Exception as e:
            raise e
