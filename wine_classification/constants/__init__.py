import os
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
ARTIFACT_DIR: str = "artifact"
PIPLELINE_NAME: str = "wineproject"

MODEL_FILE_NAME = "model.pkl"

TARGET_COLUMN = "quality_label"

FILE_NAME = "wine_quality_classification.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
RAW_DATA_FILE_PATH = os.path.join("data", FILE_NAME)
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")
PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "wine_data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2