import os
import pandas as pd


class Utility:
    def __init__(self) -> None:
        pass

    def read_data(data):
        try:
            # read dataframe
            df = pd.read_csv(data, index_col=False)
            return df

        except Exception as e:
            raise e
