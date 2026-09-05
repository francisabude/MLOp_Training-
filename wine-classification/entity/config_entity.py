import os
from datetime import datetime
from constants import *
from dataclasses import dataclass


@dataclass
class TrainPipelineConfig:
    pipeline_name = " Wine_Pipeline"
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP


training_pipeline_config = TrainPipelineConfig()

@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config, "DATA_INGESTION")