import os
import sys
from src.Titanic_Survival.components.data_ingestion import DataIngestion
from src.Titanic_Survival.components.data_transformation import Datatransformation
from src.Titanic_Survival.components.model_training import Modeltrainer
from src.Titanic_Survival.logger import logging
from src.Titanic_Survival.exception import CustomException