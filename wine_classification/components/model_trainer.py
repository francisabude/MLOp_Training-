import sys
from typing import Tuple

import mlflow
import mlflow.sklearn
import numpy as np
from neuro_mf import ModelFactory, BestModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from wine_classification.configuration.mlflow_connection import configure_mlflow_tracking
from wine_classification.constants import MLFLOW_MODEL_ARTIFACT_NAME
from wine_classification.entity.artifact_entity import (
    ClassificationMetricArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from wine_classification.entity.config_entity import ModelTrainerConfig
from wine_classification.entity.estimator import WineModel
from wine_classification.exception import WineException
from wine_classification.logger import logging
from wine_classification.utils.main_utils import load_numpy_array_data, load_object, save_object


class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        :param data_transformation_artifact: Output reference of data transformation artifact stage
        :param model_trainer_config: Configuration for model training
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
        self._grid_searched_models = []
        self._test_accuracy = 0.0

    def get_model_object_and_report(self, train: np.ndarray, test: np.ndarray) -> Tuple[BestModel, ClassificationMetricArtifact]:
        """
        Method Name :   get_model_object_and_report
        Description :   Uses neuro_mf ModelFactory to grid search every model listed in config/model.yaml,
                        keeps the best one and evaluates it on the hold-out test array.

        Output      :   Returns the neuro_mf BestModel and a ClassificationMetricArtifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Using neuro_mf to get best model object and report")
            model_factory = ModelFactory(model_config_path=self.model_trainer_config.model_config_file_path)

            x_train, y_train = train[:, :-1], train[:, -1].astype(int)
            x_test, y_test = test[:, :-1], test[:, -1].astype(int)

            # base_accuracy=0.0 lets every candidate through so the top model is always returned;
            # the expected-score check is applied explicitly in initiate_model_trainer.
            best_model_detail: BestModel = model_factory.get_best_model(X=x_train, y=y_train, base_accuracy=0.0)
            self._grid_searched_models = model_factory.grid_searched_best_model_list or []

            model_obj = best_model_detail.best_model
            y_pred = model_obj.predict(x_test)

            metric_artifact = ClassificationMetricArtifact(
                f1_score=float(f1_score(y_test, y_pred, average="weighted")),
                precision_score=float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
                recall_score=float(recall_score(y_test, y_pred, average="weighted")),
            )
            self._test_accuracy = float(accuracy_score(y_test, y_pred))
            logging.info(f"Best model: {type(model_obj).__name__} | test metrics: {metric_artifact} | "
                         f"accuracy: {self._test_accuracy:.4f}")
            return best_model_detail, metric_artifact

        except Exception as e:
            raise WineException(e, sys) from e

    def _log_to_mlflow(self, best_model_detail: BestModel, metric_artifact: ClassificationMetricArtifact,
                       train: np.ndarray, test: np.ndarray, meets_expected_score: bool) -> str:
        """
        Logs parameters, metrics, the sklearn model and the pipeline artifacts of this training run
        to the MLflow tracking server (DagsHub when configured). Returns the MLflow run id.
        """
        try:
            model_obj = best_model_detail.best_model
            with mlflow.start_run(run_name=type(model_obj).__name__) as run:
                mlflow.set_tags({
                    "stage": "model_trainer",
                    "best_model": type(model_obj).__name__,
                    "meets_expected_score": str(meets_expected_score).lower(),
                })

                mlflow.log_params({
                    "model_class": type(model_obj).__name__,
                    "model_serial_number": best_model_detail.model_serial_number,
                    "expected_f1_score": self.model_trainer_config.expected_f1_score,
                    "train_rows": int(train.shape[0]),
                    "test_rows": int(test.shape[0]),
                    "n_features": int(train.shape[1] - 1),
                })
                mlflow.log_params({f"best_{k}": v for k, v in best_model_detail.best_parameters.items()})

                mlflow.log_metrics({
                    "cv_best_score": float(best_model_detail.best_score),
                    "test_accuracy": self._test_accuracy,
                    "test_f1_weighted": metric_artifact.f1_score,
                    "test_precision_weighted": metric_artifact.precision_score,
                    "test_recall_weighted": metric_artifact.recall_score,
                })
                # Cross-validation score of every candidate model, for side-by-side comparison on DagsHub
                for searched in self._grid_searched_models:
                    mlflow.log_metric(f"cv_score_{type(searched.best_model).__name__}", float(searched.best_score))

                # cloudpickle: the default skops format rejects some estimators (e.g. KNN's KDTree types)
                mlflow.sklearn.log_model(
                    model_obj,
                    name=MLFLOW_MODEL_ARTIFACT_NAME,
                    input_example=train[:5, :-1],
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                )
                mlflow.log_artifact(self.model_trainer_config.model_config_file_path, artifact_path="config")
                mlflow.log_artifact(self.data_transformation_artifact.transformed_object_file_path,
                                    artifact_path="preprocessing")
                mlflow.log_artifact(self.model_trainer_config.trained_model_file_path, artifact_path="trained_model")

                logging.info(f"MLflow run {run.info.run_id} logged to {mlflow.get_tracking_uri()}")
                return run.info.run_id
        except Exception as e:
            raise WineException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates the model trainer steps of the training pipeline

        Output      :   Returns ModelTrainerArtifact
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            tracking_uri = configure_mlflow_tracking()
            logging.info(f"Experiment tracking configured at {tracking_uri}")

            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)

            best_model_detail, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)

            meets_expected_score = metric_artifact.f1_score >= self.model_trainer_config.expected_f1_score
            if not meets_expected_score:
                logging.warning(
                    f"Best model f1 score {metric_artifact.f1_score:.4f} is below the expected "
                    f"{self.model_trainer_config.expected_f1_score}. The model is still saved and logged "
                    "so the run can be reviewed on DagsHub."
                )

            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            wine_model = WineModel(preprocessing_object=preprocessing_obj,
                                   trained_model_object=best_model_detail.best_model)
            save_object(self.model_trainer_config.trained_model_file_path, wine_model)
            logging.info(f"Saved WineModel to {self.model_trainer_config.trained_model_file_path}")

            run_id = self._log_to_mlflow(best_model_detail, metric_artifact, train_arr, test_arr, meets_expected_score)
            logging.info(f"Training run logged to MLflow with run_id={run_id}")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise WineException(e, sys) from e


if __name__ == "__main__":
    from wine_classification.components.data_ingestion import DataIngestion
    from wine_classification.components.data_validation import DataValidation
    from wine_classification.components.data_transformation import DataTransformation
    from wine_classification.entity.config_entity import DataValidationConfig, DataTransformationConfig

    data_ingestion_artifact = DataIngestion().ingest_data()
    data_validation_artifact = DataValidation(
        data_ingestion_artifact=data_ingestion_artifact,
        data_validation_config=DataValidationConfig(),
    ).initiate_data_validation()
    data_transformation_artifact = DataTransformation(
        data_ingestion_artifact=data_ingestion_artifact,
        data_validation_artifact=data_validation_artifact,
        data_transformation_config=DataTransformationConfig(),
    ).initiate_data_transformation()
    model_trainer_artifact = ModelTrainer(
        data_transformation_artifact=data_transformation_artifact,
        model_trainer_config=ModelTrainerConfig(),
    ).initiate_model_trainer()
    print(model_trainer_artifact)
