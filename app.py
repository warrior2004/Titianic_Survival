from src.Titanic_Survival.logger import logging
from src.Titanic_Survival.exception import CustomException
from src.Titanic_Survival.components.data_ingestion import DataIngestion
from src.Titanic_Survival.components.data_transformation import Datatransformation
from src.Titanic_Survival.components.model_training import Modeltrainer
import sys

if __name__ == '__main__':
    logging.info("The execution has started")

    try:
        data_ingestion = DataIngestion()
        train_data_path,test_data_path = data_ingestion.initiate_data_ingestion()

        #data_transformation_config = DataTransformationConfig()
        data_transformation = Datatransformation()
        train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data_path,test_data_path)

        # Model trainer
        model_trainer = Modeltrainer()
        print(model_trainer.initiate_model_trainer(train_arr,test_arr))

    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)