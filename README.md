# ML Drug Classification Pipeline
![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)
## Table of Content
- [Problem Statement](#problem-statement)
- [Setup](#setup)
- [Development](#development)
- [Orchestration](#orchestration)
- [Test](#test)
- [Deployment](#deployment)
- [Monitor](#monitor)
- [CICD](#cicd)

## Problem Statement
This dataset contains information about drug classification based on patient general information and its diagnosis. Machine learning model is needed in order to predict the outcome of the drugs type that might be suitable for the patient.

## Setup
**Installation:** Clone the repository git clone https://github.com/Hsinghsudwal/ml_drug_classification.git

Set up Environment for managing libraries and running python scripts.

`conda create -n venv python==3.12 -y` and activate the environment `conda activate venv`

Install Dependencies:
 
`pip install -r requirements.txt`

## Development
**Notebook:**
Run Jupyter Notebook: On the terminal, from your main project directory.

   `cd notebook` and `jupyter lab`

Dataset exploration and perform EDA, Feature Engineering, Model Trainer and Model Evaluation. Also use Hypertuning technique such as GridSearchCV.

Build various ML models that can predict drug type. The  models that are used:

    Linear Logistic Regression
    Support Vector Classifier (SVC)
    K Neighbours
    Decision Tree
    Random Forest

## Orchestration

**Src:** From notebook to scripts

Edit script into modular code. To perform pipeline functions: which are located src/ml_drug_classification - data_loader, data_validation, data_transformation,model_trainer, model evaluation, model_staging and model_production. To run scripts from main project directory:

`python run.py`

This script will output verious files and model to save into output and use in deployment

**Model Experiments:**
Mlflow: using tracking runs and its storage allow different parameters to register best model and put that model to production.

mlflow tracking server:

    mlflow server --backend-store-uri   sqlite:///mlruns.db --default-artifact-root artifacts -p 8080

    # Set our tracking server uri for logging mlflow.set_tracking_uri (uri="http://127.0.0.1:8080")

    # Create a new MLflow Experiment 
    
    mlflow.set_experiment("experiment_name")

    Start an MLflow run

        with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params)

        # Log the loss metric
        mlflow.log_metric("accuracy", accuracy)

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", "Basic model")

        # Infer the model signature
        signature = infer_signature(X_train, lr.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model_path",
        signature=signature,
        input_example=X_train,
    )
Load the model back for predictions as a generic Python Function model.

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    predictions = loaded_model.predict(test)

## Test

With pytest to test if model exists on mlflow staging, perform simply model prediction to get accuracy and evaluate.
`cd test` and  `pytest test.py`

1. Loading Data: The test_data function loads the test data (X_test and y_test).
2. Preprocessing Data: The process function transforms the test data (filling missing values, encoding categorical variables, scaling numerical values).
3. Loading Model: The test_model function loads the most recent version of the model from MLflow's "Staging" stage.
4. Making Predictions: The model makes predictions on the preprocessed test data.
5. Evaluating Model: The test_evaluate function calculates the model's accuracy and checks if it meets the required threshold of 90%. Based on the accuracy, it decides whether the model is ready for production or needs retraining.

To run locally via 

```bash
python test.py
```

## Deployment
Created web application built using Streamlit that classifies a drug type based on user input. The app uses a pre-trained machine learning model to predict the drug type based on the following user inputs:

**Installation:**
1. From the terminal cd to working directory 
`cd deployment`

2. Install dependencies using:
```bash
    pip install -r requirements.txt
```
3. Run the app using Streamlit:
```bash
    streamlit run app.py
```

* Streamlit provides the user interface for inputting data.
* The pre-trained machine learning model is used to predict the drug type based on user input.
* Data preprocessing (handling missing values, encoding, and scaling) is done before making the prediction.
* The app outputs the predicted drug type based on the input data provided by the user.

The application is an example of how to deploy a machine learning model into a user-friendly web interface with Streamlit, where users can input data and get real-time predictions.

**Docker**
1. Build the Docker Image:

```bash
    docker build -t app .
```

2. Run the Docker Image Locally:

```bash
    docker run -p 8501:8501 app
```
`http://localhost:8501.`

## Monitor

Create app is a drug classification monitoring and model drift detection system that integrates with Streamlit, Prometheus and alerts. The primary objective is to monitor the performance of a machine learning model, detect when the model's performance deteriorates (model drift), and trigger alerts. Additionally, the system exposes performance metrics using Prometheus for monitoring purposes.

## Installation:

From terminal to directory
    `cd monitor`

2. Install dependencies using:

    `pip install -r requirements.txt`

3. Run code locally via `python app.py` and with streamlit via  `streamlit run app.py`

Build and Start the Services: From the project directory, run:

```bash
docker-compose up --build
```


* Services:
    Streamlit: http://localhost:8501.
    Prometheus: http://localhost:8000.
    Grafana: http://localhost:3000 using admin as the username and password.

## Next Step
* CI/CD
* Cloud



