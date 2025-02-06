## Drug Classification Monitoring App
This script app is a drug classification monitoring and model drift detection system that integrates with Streamlit, Prometheus and alerts. The primary objective is to monitor the performance of a machine learning model, detect when the model's performance deteriorates (model drift), and trigger alerts. Additionally, the system exposes performance metrics using Prometheus for monitoring purposes. The script performs the following main tasks:

**Key Components:**
Data Preprocessing: Imputation, encoding, scaling.
Model Drift Detection: Monitors accuracy and classification report.
Logging: Logs drift events.
Prometheus Metrics: Collects and exposes metrics for monitoring.
Streamlit: Provides a real-time user interface to display results.

1. Load Data: A dataset containing drug-related information is loaded using pandas. The data is read from a CSV file.

2. Split the data into training and test datasets.

3. Data Validation (Data Drift Detection): Before processing the data, the script performs a KS (Kolmogorov-Smirnov) test to detect whether the distribution of features has significantly changed between the training and test datasets. If there’s a significant difference (p-value < 0.05), it logs a data drift error.

4. Data Preprocessing: The training and testing datasets are processed to handle categorical and numerical columns. Which involves: 

* Imputing missing values.
* OrdinalEncoder: categorical.
* StandardScaler: numerical. This processed data ready for model evaluation.

5. Model Drift Detection: The code checks the performance of the model on the test set. If the model accuracy drops below 90%, it is considered a drift. The model's accuracy is logged, and a Prometheus metric is updated to monitor this performance. If drift is detected:
* The model accuracy is displayed via Streamlit.
* A drift alert is logged and displayed with a classification report.

6. Prometheus Monitoring: A Prometheus metrics server is started to expose the model's accuracy and drift status on a specified port (8000). This allows for continuous monitoring and tracking of the model's performance over time.

7. Logging and Alerts: The code uses Python's logging library to log any significant events like model drift. The model's drift status and classification reports are logged for further analysis.

8. Streamlit App: The script runs as a Streamlit app where users can interactively view the model’s performance. It continuously checks for drift, showing real-time updates on the model's accuracy and performance metrics.

## Installation:
**Steps:**
1. Clone the repository.
`cd monitor` and `conda activate venv`
2. Install dependencies using:

    `pip install -r requirements.txt`

3. Run code locally via `python app.py` and with streamlit via  `streamlit run app.py`

Create a directory for the project
```
prod_model.py
model.joblib
Dockerfile
docker-compose.yml
prometheus.yml
app.py
requirements.txt
```

Build and Start the Services: From the project directory, run:

```bash
docker-compose up --build
```
* Services:
    Streamlit: Access the Streamlit app at http://localhost:8501.
    Prometheus: Access Prometheus at http://localhost:8000.
    Grafana: Access Grafana at http://localhost:3000 using admin as the username and password.


