import pandas as pd
from src.utility import *
from src.constants import *
import os

# from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib


class ModelTrainer:

    def __init__(self) -> None:
        pass

    def model_trainer(self, X_train, y_train):
        try:

            # clf = RandomForestClassifier()
            param_grids = {
                "n_estimators": [50, 100, 200],
                "max_depth": [10, 20, 30],
                "min_samples_split": [2, 5, 10],
            }
            # Define the pipeline
            # pipeline = Pipeline(
            #     steps=[

            #         ("smote", SMOTE()),
            #         ("classifier", clf),
            #     ])

            # GridSearchCV
            grid_search = GridSearchCV(
                RandomForestClassifier(),
                param_grid=param_grids,
                cv=3,
                n_jobs=-1,
                verbose=1,
            )

            # Fit the model
            grid_search.fit(X_train, y_train)

            best_params = grid_search.best_params_
            # print(f"Best parameters for {clf}: {best_params}")

            best_model = grid_search.best_estimator_

            model_path = os.path.join(OUTPUT, MODEL_FOLDER)
            os.makedirs(model_path, exist_ok=True)

            model_filename = os.path.join(model_path, BEST_MODEL_NAME)
            joblib.dump((best_model, best_params), model_filename)

            print("Model Trainer Completed")
            return best_model, best_params

        except Exception as e:
            raise e
