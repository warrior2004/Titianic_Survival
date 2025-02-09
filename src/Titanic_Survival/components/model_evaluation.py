import os
import sys
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from src.Titanic_Survival.exception import CustomException
from src.Titanic_Survival.logger import logging
from src.Titanic_Survival.utils import load_object

class ModelEvaluator:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()
    
    def load_model(self):
        try:
            model = load_object(self.model_path)
            logging.info("Model loaded successfully.")
            return model
        except Exception as e:
            raise CustomException(e, sys)
    
    def eval_metrics(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        return accuracy, precision, recall, f1
    
    def evaluate(self, test_array):
        try:
            logging.info("Evaluating the trained model on test data.")
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            y_pred = self.model.predict(X_test)

            accuracy, precision, recall, f1 = self.eval_metrics(y_test, y_pred)
            logging.info(f"Model Evaluation Metrics: Accuracy={accuracy}, Precision={precision}, Recall={recall}, F1 Score={f1}")

            # Log metrics using MLflow
            with mlflow.start_run():
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1", f1)
                mlflow.sklearn.log_model(self.model, "model")

            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "classification_report": classification_report(y_test, y_pred),
                "confusion_matrix": confusion_matrix(y_test, y_pred)
            }
        
        except Exception as e:
            raise CustomException(e, sys)
