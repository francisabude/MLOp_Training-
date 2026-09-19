import json
import re
import sys

import pandas as pd
from pandas import DataFrame
from evidently import Report
from evidently.presets import DataDriftPreset


from wine_classification.entity.config_entity import DataValidationConfig
from wine_classification.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from wine_classification.logger import logging
from wine_classification.exception import WineException
from wine_classification.constants import SCHEMA_FILE_PATH
from wine_classification.utils.main_utils import read_yaml_file, write_yaml_file


class DataValidation:
        def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
            """
            :param data_ingestion_artifact: Output reference of data ingestion artifact stage
            :param data_validation_config: configuration for data validation
            """
            try:
                self.data_ingestion_artifact = data_ingestion_artifact
                self.data_validation_config = data_validation_config
                self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
            except Exception as e:
                raise WineException(e,sys)

        def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
            """
            function to validate our columns
            """
            try:
                schema_columns = [
                    list(col.keys())[0] if isinstance(col, dict) else col
                    for col in self._schema_config["columns"]
                ]
                status = len(dataframe.columns) == len(schema_columns)
                logging.info(f"Is required columns present: [{status}]")
                return status
            except Exception as e:
                raise WineException(e, sys)
            
        def is_column_exist(self, df: DataFrame) -> bool:
            """
            function to find out whether the name of a column has changed
            """

            try:
                dataframe_columns = df.columns
                missing_numerical_columns = []
                missing_categorical_columns = []

                for column in self._schema_config["numerical_columns"]:
                    col_name = list(column.keys())[0] if isinstance(column, dict) else column
                    if col_name not in dataframe_columns:
                        missing_numerical_columns.append(col_name)

                if len(missing_numerical_columns) > 0:
                    logging.info(f"Missing numerical column: {missing_numerical_columns}")

                if len(missing_numerical_columns) == 0:
                    logging.info(f"There are no missing numerical columns.")

                for column in self._schema_config["categorical_columns"]:
                    col_name = list(column.keys())[0] if isinstance(column, dict) else column
                    if col_name not in dataframe_columns:
                        missing_categorical_columns.append(col_name)

                if len(missing_categorical_columns) > 0:
                    logging.info(f"Missing categorical column: {missing_categorical_columns}")

                if len(missing_categorical_columns) == 0:
                    logging.info(f"There are no missing categorical columns.")

                return False if len(missing_categorical_columns) > 0 or len(missing_numerical_columns) > 0 else True
            except Exception as e:
                raise WineException(e, sys) from e
            
        @staticmethod
        def read_data(file_path) -> DataFrame:
            try:
                return pd.read_csv(file_path)
            except Exception as e:
                raise WineException(e, sys)
            
            
        def detect_dataset_drift(self, reference_df: DataFrame, current_df: DataFrame) -> bool:
            try:
                report = Report([DataDriftPreset()])
                snapshot = report.run(reference_data=reference_df, current_data=current_df)
                json_report = json.loads(snapshot.json())

                write_yaml_file(
                    file_path=self.data_validation_config.drift_report_file_path,
                    content=json_report,
                )

                metrics = json_report.get("metrics", [])

               
                def mid(m):
                    return m.get("metric_id") or m.get("metric_name") or ""

                drifted_col_metric = next(
                    (m for m in metrics if "DriftedColumnsCount" in mid(m)),
                    None,
                )
                if drifted_col_metric is None:
                    raise ValueError(
                        f"DriftedColumnsCount not found. Available: {[mid(m) for m in metrics]}"
                    )

                value = drifted_col_metric.get("value", {})
                n_drifted = value.get("count", 0) if isinstance(value, dict) else int(value)
                share = value.get("share") if isinstance(value, dict) else None

                # Pull drift_share threshold out of the metric_id signature, fallback 0.5
                m = re.search(r"drift_share\s*=\s*([0-9.]+)", mid(drifted_col_metric))
                drift_share = float(m.group(1)) if m else 0.5

                value_drift_metrics = [m for m in metrics if "ValueDrift" in mid(m)]
                n_features = len(value_drift_metrics)

                if share is None:
                    share = (n_drifted / n_features) if n_features > 0 else 0.0

                drift_status = share >= drift_share

                logging.info(
                    f"{int(n_drifted)}/{n_features} columns drifted "
                    f"(share={share:.3f}, threshold={drift_share}). Dataset drift: {drift_status}"
                )
                return drift_status

            except Exception as e:
                raise WineException(e, sys) from e
            
        def initiate_data_validation(self) -> DataValidationArtifact:
            """
            Method Name :   initiate_data_validation
            Description :   This method initiates the data validation component for the pipeline
            
            Output      :   Returns bool value based on validation results
            On Failure  :   Write an exception log and then raise an exception
            """

            try:
                validation_error_msg = ""
                logging.info("Starting data validation")
                train_df, test_df = (DataValidation.read_data(file_path=self.data_ingestion_artifact.trained_file_path),
                                    DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path))

                status = self.validate_number_of_columns(dataframe=train_df)
                logging.info(f"All required columns present in training dataframe: {status}")
                if not status:
                    validation_error_msg += f"Columns are missing in training dataframe."

                status = self.validate_number_of_columns(dataframe=test_df)
                logging.info(f"All required columns present in testing dataframe: {status}")
                if not status:
                    validation_error_msg += f"Columns are missing in test dataframe."

                status = self.is_column_exist(df=train_df)
                if not status:
                    validation_error_msg += f"Columns are missing in training dataframe."

                status = self.is_column_exist(df=test_df)
                if not status:
                    validation_error_msg += f"columns are missing in test dataframe."

                validation_status = len(validation_error_msg) == 0

                if validation_status:
                    logging.info("drift Started..")
                    drift_status = self.detect_dataset_drift(train_df, test_df)
                    if drift_status:
                        logging.info(f"Drift detected.")
                        validation_error_msg = "Drift detected"
                    else:
                        validation_error_msg = "Drift not detected"
                    logging.info("drift ended..")
                else:
                    logging.info(f"Validation_error: {validation_error_msg}")
                    

                data_validation_artifact = DataValidationArtifact(
                    validation_status=validation_status,
                    message=validation_error_msg,
                    drift_report_file_path=self.data_validation_config.drift_report_file_path
                )
                logging.info(f"Data validation artifact: {data_validation_artifact}")
                return data_validation_artifact
            except Exception as e:
                raise WineException(e, sys) from e
        
if __name__ == "__main__":
    from wine_classification.components.data_ingestion import DataIngestion

    data_ingestion_artifact = DataIngestion().ingest_data()
    data_validation = DataValidation(
        data_ingestion_artifact=data_ingestion_artifact,
        data_validation_config=DataValidationConfig(),
    )
    data_validation_artifact = data_validation.initiate_data_validation()
    print(data_validation_artifact)