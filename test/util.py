import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


def process(test):

    cols = test.columns
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
