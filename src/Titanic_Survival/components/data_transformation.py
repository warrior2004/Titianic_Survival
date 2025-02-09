import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler,MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.Titanic_Survival.exception import CustomException
from src.Titanic_Survival.logger import logging
from src.Titanic_Survival.utils import save_object
from feature_engine.outliers.winsorizer import Winsorizer
from feature_engine.encoding import CountFrequencyEncoder
import os

@dataclass
class DatatransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class Datatransformation:
    def __init__(self):
        self.data_transformation_config = DatatransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            # age_pipeline
            age_pipe = Pipeline(steps=[
            ('impute',SimpleImputer(strategy='median')),
            ('outliers',Winsorizer(capping_method='gaussian',fold=3)),
            ('scale',StandardScaler())
            ])

            # fare pipeline
            fare_pipe = Pipeline(steps=[
            ('outliers',Winsorizer(capping_method='iqr',fold=1.5)),
            ('scale',StandardScaler())
            ])

            # embarked_pipeline
            embarked_pipe = Pipeline(steps=[
            ('impute',SimpleImputer(strategy='most_frequent')),
            ('count_encode',CountFrequencyEncoder(encoding_method='count')),
            ('scale',MinMaxScaler())
            ])

            logging.info("Created pipeline for age,fare and embarked features")

            preprocessor = ColumnTransformer(transformers=[
            ('age',age_pipe,['age']),
            ('fare',fare_pipe,['fare']),
            ('embarked',embarked_pipe,['embarked']),
            ('sex',OneHotEncoder(sparse_output=False,handle_unknown='ignore'),['sex']),
            ('family',MinMaxScaler(),['family'])
            ],remainder='passthrough',n_jobs=-1,force_int_remainder_cols=False)
            
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
    def clean_data(self,df):
        # columns to drop
        columns_to_drop = ['passengerid','name','ticket','cabin']
        return (
        df
        .rename(columns=str.lower)
        .drop(columns=columns_to_drop)
        .assign(
            family = lambda df_ : df_['sibsp'] + df_['parch']
        )
        .drop(columns=['sibsp','parch'])
        )
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train and test paths")

            preprocessing_obj = self.get_data_transformer_object()
            target_column = 'survived'

            # Clean the data
            train_df = self.clean_data(train_df)
            test_df = self.clean_data(test_df)

            # Divide the train dataset inot independent and dependent feature
            input_train_df = train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df = train_df[target_column]

            # Divide the train dataset inot independent and dependent feature
            input_test_df = test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info("Applying Preprocessing into train and test data")

            input_features_train_arr = preprocessing_obj.fit_transform(input_train_df)
            input_features_test_arr = preprocessing_obj.transform(input_test_df)

            train_arr = np.c_[
                input_features_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_features_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return (

                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)