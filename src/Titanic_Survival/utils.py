import os
import sys
from src.Titanic_Survival.exception import CustomException
from src.Titanic_Survival.logger import logging
from sqlalchemy import create_engine
import pickle
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")

# Create a SQLAlchemy engine
engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}/{db}')

def read_sql_data():
    logging.info("Reading SQL database started")
    try:
        df = pd.read_sql_table('titanic',engine)
        print(df.head())
        return df
    except Exception as e:
        raise CustomException(e,sys)
    
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)