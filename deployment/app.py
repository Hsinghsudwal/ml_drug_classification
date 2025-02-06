import streamlit as st
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Load the pre-trained model
model = joblib.load("app_model.joblib")


def predict_user_input(Age, Sex, BP, Cholesterol, Na_to_K):
    # Create input dataframe
    input_data = pd.DataFrame(
        {
            "Age": [Age],
            "Sex": [Sex],
            "BP": [BP],
            "Cholesterol": [Cholesterol],
            "Na_to_K": [Na_to_K],
        }
    )

    # Define categorical and numerical columns
    cat_col = ["Sex", "BP", "Cholesterol"]
    num_col = ["Age", "Na_to_K"]

    # Preprocessing pipeline for categorical and numerical columns
    preprocess = ColumnTransformer(
        transformers=[
            (
                "encoder",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OrdinalEncoder()),
                    ]
                ),
                cat_col,
            ),
            (
                "num_transform",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_col,
            ),
        ],
        remainder="passthrough",
    )

    # Preprocess user input
    preprocessed_input = preprocess.fit_transform(input_data)
    preprocessed_input_df = pd.DataFrame(preprocessed_input, columns=cat_col + num_col)

    # Prediction
    predicted_drug = model.predict(preprocessed_input_df)[0]
    return f"Predicted Drug: {predicted_drug}"


# Streamlit UI for user interaction
st.title("Drug Classification")
st.markdown("Enter the details to correctly identify the drug type:")

# Input widgets for user interaction
Age = st.number_input("Age", min_value=15, max_value=74, value=15)
Sex = st.radio("Sex", ["M", "F"])
BP = st.radio("Blood Pressure", ["HIGH", "LOW", "NORMAL"])
Cholesterol = st.radio("Cholesterol", ["HIGH", "NORMAL"])
Na_to_K = st.text_input("Na_to_K Ratio", value="0")

# Display entered values
st.write(Age, Sex, BP, Cholesterol, Na_to_K)

# When the "Predict Drug" button is clicked
if st.button("Predict Drug"):
    # Call the prediction function with user inputs
    result = predict_user_input(Age, Sex, BP, Cholesterol, Na_to_K)
    st.write(result)
