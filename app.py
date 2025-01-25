from src.Titanic_Survival.logger import logging
from src.Titanic_Survival.exception import CustomException
from src.Titanic_Survival.components.data_ingestion import DataIngestion
import sys

if __name__ == '__main__':
    logging.info("The execution has started")

    try:
        data_ingestion = DataIngestion()
        train_data_path,test_data_path = data_ingestion.initiate_data_ingestion()

    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)