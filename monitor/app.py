import pandas as pd
import numpy as np
import os
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import ks_2samp
import logging
import streamlit as st
from prometheus_client import start_http_server, Gauge
import time

# Prometheus Metrics
MODEL_ACCURACY = Gauge("model_accuracy", "Current model accuracy")
MODEL_DRIFT_ALERT = Gauge("model_drift_alert", "Alert for model drift")

# Set up logging to log model drift events
logging.basicConfig(
    filename="model_drift.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_data(path):
    """Loads dataset from the specified path."""
    data = pd.read_csv(path)
    return data


def split_data(data):
    """Splits the dataset into training and test data (70/30)."""
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)
    return train_data, test_data


def data_validator(train_data, test_data):
    """Validates data drift using KS test."""
    data_results = {}
    for feature in train_data.columns:
        ks_stat, p_value = ks_2samp(train_data[feature], test_data[feature])
        data_results[feature] = p_value
        if p_value < 0.05:
            return "Error drift data"
    return "Data validated"


def data_process(train_data, test_data):

    xtrain_data = train_data.drop(columns=["Drug"], axis=1)
    xtest_data = test_data.drop(columns=["Drug"], axis=1)
    y_train = train_data["Drug"]
    y_test = test_data["Drug"]

    cat_col = [1, 2, 3]
    num_col = [0, 4]

    transform = ColumnTransformer(
        transformers=[
            (
                "encoder",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OrdinalEncoder()),
                    ]
                ),
                cat_col,
            ),
            (
                "num_transform",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_col,
            ),
        ],
        remainder="passthrough",
    )

    xtrain_preprocessed = transform.fit_transform(xtrain_data)
    xtest_preprocessed = transform.transform(xtest_data)

    # Convert transformed data back into DataFrame for easy analysis
    X_train = pd.DataFrame(xtrain_preprocessed, columns=xtrain_data.columns)
    X_test = pd.DataFrame(xtest_preprocessed, columns=xtest_data.columns)

    return X_train, X_test, y_train, y_test


def load_model():
    """Loads the pre-trained model."""
    loaded_model = joblib.load(open("app_model.joblib", "rb"))
    return loaded_model


def models_drift(model, xtest, ytest):
    """Checks for model drift by evaluating performance on the test set."""
    df = pd.DataFrame(xtest)
    y_pred = model.predict(df)
    accuracy = round(accuracy_score(ytest, y_pred), 2)
    report = classification_report(ytest, y_pred)

    # Update Prometheus metric for model accuracy
    MODEL_ACCURACY.set(accuracy)

    # If model accuracy drops below threshold (0.9), consider it drift
    if accuracy < 0.9:
        MODEL_DRIFT_ALERT.set(1)
        handle_drift_alert(accuracy, report)
    else:
        MODEL_DRIFT_ALERT.set(0)

    return accuracy, report


def handle_drift_alert(accuracy, report):
    """Handles drift alert by logging to a file and showing Streamlit warning."""
    # Log to a file
    logging.info(
        f"Model drift detected! Accuracy: {accuracy}\nClassification Report: {report}"
    )

    # Show a visual alert in Streamlit
    st.warning(f"Model drift detected! Current accuracy: {accuracy}")
    st.text(f"Classification Report:\n{report}")


def start_prometheus_server():
    """Starts the Prometheus metrics server."""
    start_http_server(8000)  # Expose metrics on port 8000


def main():
    """Main function that ties everything together and runs the Streamlit app."""
    st.title("Drug Classification Monitoring")
    path = r"../data_source/drug.csv"
    data = load_data(path)

    # Split data into training and testing sets
    train, test = split_data(data)

    # Validate data for drift
    data_status = data_validator(train, test)
    if data_status != "Data validated":
        st.write("Data drift detected. Exiting...")
        return

    # Process the data (imputation, encoding, scaling)
    X_train, X_test, y_train, y_test = data_process(train, test)

    # Load pre-trained model
    loaded_model = load_model()

    # Start Prometheus server for metrics collection
    start_prometheus_server()

    # Loop for continuous drift monitoring
    while True:
        accuracy, report = models_drift(loaded_model, X_test, y_test)
        st.write(f"Model Accuracy: {accuracy}")
        st.write(f"Model Report: {report}")
        time.sleep(10)  # Check every minute


if __name__ == "__main__":
    main()
