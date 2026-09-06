import os
import pandas as pd
import sys
import yaml
from pandas import DataFrame
import numpy as np
from wine_classification.exception import WineException
from wine_classification.logger import logging




def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise WineException(e, sys) from e
    
@staticmethod
def read_data(file_path) -> DataFrame:
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise WineException(e, sys)

def drop_columns(df: DataFrame, cols: list)-> DataFrame:
    """
    drop the columns form a pandas DataFrame
    df: pandas DataFrame
    cols: list of columns to be dropped
    """
    logging.info("Entered drop_columns methon of utils")
    try:
        df = df.drop(columns=cols, axis=1)
        logging.info("Exited the drop_columns method of utils") 
        return df
    except Exception as e:
        raise WineException(e, sys) from e