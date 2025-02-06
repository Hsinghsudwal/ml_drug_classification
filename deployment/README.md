## Drug Classification Streamlit App
Created web application built using Streamlit that classifies a drug type based on user input. The app uses a pre-trained machine learning model to predict the drug type based on the following user inputs:

* Age: The age of the person.
* Sex: The sex of the person (Male or Female).
* Blood Pressure: The person's blood pressure (High, Low, Normal).
* Cholesterol: The person's cholesterol level (High, Normal).
* Na_to_K Ratio: The sodium to potassium ratio.

-Interactive UI for input using Streamlit.

-The app applies preprocessing techniques such as imputation and encoding on the input data before passing it to the model for prediction.

**Installation:**
1. Clone the repository.
`cd deployment` and `conda activate venv`
2. Install dependencies using:
```bash
pip install -r requirements.txt
```
3. Run the app using Streamlit:
```bash
streamlit run app.py
```
### Docker:

**Instructions** to Build, Run, and Push to a Docker Repository
Steps:
1. Build the Docker Image:

Navigate to the directory containing the Dockerfile and run the following command to build the image:

```bash
docker build -t hsinghsudwal/drug-classification-app .
```

2. Run the Docker Image Locally:

After the build process, run the container to make sure it works correctly:

```bash
docker run -p 8501:8501 hsinghsudwal/drug-classification-app
```
`access the app locally at http://localhost:8501.`


3. Push the Docker Image to a Repository:

Login docker hub account:

```bash
docker login
```
`make sure namespace matches user-name`

```bash
docker push hsinghsudwal/drug-classification-app
```
