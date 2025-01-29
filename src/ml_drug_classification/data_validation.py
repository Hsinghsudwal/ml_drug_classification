import pandas as pd
from scipy.stats import ks_2samp
from src.utility import *
from src.constants import *
import os
import json


class DataValidation:

    def __init__(self) -> None:
        pass

    def data_validation(
        self, train_data, test_data):
        try:

            data_results = {}
            for feature in train_data.columns:
                ks_stat, p_value = ks_2samp(train_data[feature], test_data[feature])
                data_results[feature] = p_value
                
                if p_value < THRESHOLD:
                    return "Error drift data"
                
                else:
                    validate_data_path = os.path.join(OUTPUT,VALIDATE_FOLDER)
                    os.makedirs(validate_data_path,exist_ok=True)
                    # save train test to csv
                    train_data.to_csv(os.path.join(validate_data_path,str(TRAIN_DATA_FILE)),index=False)
                    test_data.to_csv(os.path.join(validate_data_path,str(TEST_DATA_FILE)),index=False)
                    report_file=os.path.join(validate_data_path,REPORT_JSON)

                    with open(report_file,'w') as f_in:
                        json.dump(data_results,f_in,indent=4)

                    print("Data validation completed")
                    return train_data, test_data

        except Exception as e:
            raise e