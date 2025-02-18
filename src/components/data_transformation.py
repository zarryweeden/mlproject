
import sys
from dataclasses import dataclass
import os
from src.utilis import save_object
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str= os.path.join('artifacts',"preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_trasformation_config= DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns= ['writing score','reading score']
            categorical_columns=[
                'gender',
                'race/ethnicity',
                'parental level of education',
                'lunch',
                'test preparation course'
            ]
            num_pipeline= Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("One hot_encoder",OneHotEncoder()),
                    ("Scaler",StandardScaler(with_mean=False))

                ]
            )
            logging.info("Numerical scaling:{categorical_columns}")

            logging.info("Categorical encoding completed{numerical_columns}")

            preprocessor= ColumnTransformer([
                ("num_pipeline",num_pipeline,numerical_columns),
                ("Cat_columns",cat_pipeline,categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException (e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df= pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Reading train and test data completed") 

            logging.info("Obtaining preprocessing object ")
            preprocessing_obj=self.get_data_transformer_object()

            target_column_name= 'math score'
            numerical_columns= ["writing score","reading score"]

            input_feature_train_df= train_df.drop(columns=[target_column_name])
            target_feature_train_df= train_df[target_column_name]

            input_feature_test_df= test_df.drop(columns=[target_column_name])
            target_feature_test_df= test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training  dataframe and testing dataframe"
            )
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr= np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr= np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]
            logging.info(f"Saved preprocessing object")

            save_object(
                file_path=self.data_trasformation_config.preprocessor_obj_file_path,
                obj= preprocessing_obj)
                
            return(
                train_arr,test_arr,self.data_trasformation_config.preprocessor_obj_file_path,
            )
            
                 
            


 

        except Exception as e:
            raise CustomException(e,sys)


