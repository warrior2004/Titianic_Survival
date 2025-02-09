import os
import sys
from dataclasses import dataclass
from urllib.parse import urlparse
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from src.Titanic_Survival.exception import CustomException
from src.Titanic_Survival.logger import logging
from src.Titanic_Survival.utils import save_object,evaluate_models
import mlflow
import mlflow.sklearn
import dagshub

@dataclass
class ModeltrainerConfig:
    trained_model_path = os.path.join("artifacts","model.pkl")

class Modeltrainer:
    def __init__(self):
        self.model_trainer_config = ModeltrainerConfig()

    def eval_metrics(self,actual,pred):
        accuracy = accuracy_score(actual,pred)
        precision = precision_score(actual,pred,average='weighted')
        recall = recall_score(actual,pred,average='weighted')
        f1 = f1_score(actual,pred,average='weighted')
        return accuracy,recall,precision,f1
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestClassifier(),
                "Logistic Regression" : LogisticRegression(),
                "XGB Classifier" : XGBClassifier(),
                "Adaboost Classifier" : AdaBoostClassifier()
            }
            model_params = {
                'Random Forest': {
                'n_estimators': [80],
                'max_depth': [5],
                'random_state': [42]
                },
                'Logistic Regression': {
                'penalty': None,
                'C': [1],
                'random_state': [42]
                },
                'AdaBoost': {
                'n_estimators': [120],
                'learning_rate': [0.3],
                'random_state': [42]
                },
                'XGBoost': {
                'n_estimators': [120],
                'max_depth': [5],
                'learning_rate': [0.3],
                'random_state': [42]
                }
            }
            model_report:dict=evaluate_models(X_train,y_train,X_test,y_test,models,model_params)

             ## To get best model score from dict
            best_model_score = max(sorted(list(model_report.values())))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            print("This is the best model:")
            print(best_model_name)

            model_names =list(model_params.keys())
            actual_model = ""
            for model in model_names:
                if best_model_name == model:
                    actual_model = actual_model + model
            
            best_params = model_params[actual_model]

            # Initialize DAGsHub repository
            dagshub.init(repo_owner='warrior2004', repo_name='Titanic_Survival', mlflow=True)

            # Configure MLflow
            mlflow.set_registry_uri("https://dagshub.com/warrior2004/Titianic_Survival.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # mlflow
            with mlflow.start_run():
                predicted_qualities = best_model.predict(X_test)
                (accuracy, precision, recall,f1) = self.eval_metrics(y_test, predicted_qualities)
                mlflow.log_params(best_params)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1", f1)


                # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(best_model, "model", registered_model_name=actual_model)
                else:
                    mlflow.sklearn.log_model(best_model, "model")
                    
                mlflow.autolog()

            logging.info(f"Best model found for training and testing data")

            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            return  accuracy

        except Exception as e:
            raise CustomException(e,sys)
            
