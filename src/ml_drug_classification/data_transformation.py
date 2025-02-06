import pandas as pd
from src.utility import *
from src.constants import *
import os
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder


class DataTransformation:

    def __init__(self) -> None:
        pass

    def data_transformation(self, train_data, test_data):
        try:

            xtrain_data = train_data.drop(columns=["Drug"], axis=1)
            xtest_data = test_data.drop(columns=["Drug"], axis=1)
            y_train = train_data["Drug"]
            y_test = test_data["Drug"]

            cat_col = [1, 2, 3]
            num_col = [0, 4]

            # preprocess
            transform = ColumnTransformer(
                transformers=[
                    (
                        "encoder",
                        Pipeline(
                            steps=[
                                (
                                    "imputer",
                                    SimpleImputer(strategy="most_frequent"),
                                ),  # Impute missing categorical values
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
                                ),  # Impute missing numeric values
                                ("scaler", StandardScaler()),
                            ]
                        ),
                        num_col,
                    ),
                ],
                remainder="passthrough",
            )

            # fit and transform train and test
            xtrain_preprocessed = transform.fit_transform(xtrain_data)
            xtest_preprocessed = transform.transform(xtest_data)

            # Convert data to DataFrame for saving to CSV
            X_train = pd.DataFrame(xtrain_preprocessed, columns=xtrain_data.columns)
            X_test = pd.DataFrame(xtest_preprocessed, columns=xtest_data.columns)

            # print(X_train)
            # print(X_test)

            transformer_path = os.path.join(OUTPUT, TRANSFORMATION_FOLDER)
            os.makedirs(transformer_path, exist_ok=True)
            X_train.to_csv(os.path.join(transformer_path, str(X_TRAIN_PROCESS_DATA)))
            X_test.to_csv(os.path.join(transformer_path, str(X_TEST_PROCESS_DATA)))

            y_train.to_csv(os.path.join(transformer_path, str(Y_TRAIN_PROCESS_DATA)))
            y_test.to_csv(os.path.join(transformer_path, str(Y_TEST_PROCESS_DATA)))

            print("Data Transformation Completed")
            return X_train, X_test, y_train, y_test

        except Exception as e:
            raise e
