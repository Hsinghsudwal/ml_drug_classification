import os
from src.ml_drug_classification.data_loader import DataLoader
from src.ml_drug_classification.data_validation import DataValidation
from src.ml_drug_classification.data_transformation import DataTransformation
from src.ml_drug_classification.model_trainer import ModelTrainer

def pipeline():
        
        path=r"data_source/drug.csv"
        
        data_loader=DataLoader()
        train_data,test_data=data_loader.dataLoader(path)

        data_validate=DataValidation()
        train_data_vali,test_data_vali=data_validate.data_validation(train_data,test_data)

        data_transformer=DataTransformation()
        X_train, X_test, y_train, y_test=data_transformer.data_transformation(train_data_vali,test_data_vali)

        model_train=ModelTrainer()
        model=model_train.model_trainer(X_train, X_test, y_train, y_test)
