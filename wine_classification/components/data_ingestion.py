import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from wine_classification.entity.artifact_entity import DataIngestionArtifact
from wine_classification.entity.config_entity import DataIngestionConfig
from wine_classification.exception import WineException
from wine_classification.logger import logging


class DataIngestion:
    def __init__(self, config: DataIngestionConfig=DataIngestionConfig()):
        """
        :param data_ingestion_config: configuration for data ingestion
        """
        try:
            self.config = config
            logging.info(f"DataIngestion initialized with config: {self.config}")
        except Exception as e:
            raise Exception(e, sys)
        
        

    def ingest_data(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion process...")

            # 1. Ensure directories exist
            os.makedirs(os.path.dirname(self.config.feature_store_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.config.training_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.config.testing_file_path), exist_ok=True)

            # 2. Load the CSV file
            df = pd.read_csv("data/wine_quality_classification.csv")  
            logging.info(f"Loaded data with shape: {df.shape}")

            # 3. Save the raw data to the feature store path
            df.to_csv(self.config.feature_store_file_path, index=False)
            logging.info(f"Raw data saved to feature store at: {self.config.feature_store_file_path}")

            # 4. Train-test split
            train_df, test_df = train_test_split(
                df,
                test_size=self.config.train_test_split_ratio,
                random_state=42
            )
            logging.info("Train-test split completed.")

            # 5. Save the split files
            train_df.to_csv(self.config.training_file_path, index=False)
            test_df.to_csv(self.config.testing_file_path, index=False)
            logging.info(f"Training data saved at: {self.config.training_file_path}")
            logging.info(f"Testing data saved at: {self.config.testing_file_path}")

            # 6. Return an artifact with paths
            artifact = DataIngestionArtifact(
               feature_store_file_path=self.config.feature_store_file_path,
                trained_file_path=self.config.training_file_path,
                test_file_path=self.config.testing_file_path
            )
            logging.info(f"Data ingestion artifact created: {artifact}")

            return artifact

        except Exception as e:
            raise Exception(e, sys) 

if __name__ == "__main__":
    X = DataIngestion().ingest_data()
    print(X)