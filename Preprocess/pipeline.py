from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import pandas as pd

import joblib
from typing import Tuple, List

import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from config.core import config

class TimeTransformer(BaseEstimator, TransformerMixin):
    """ Normalize time, e.g. 120000 => 0.5"""
    def __init__(self, variables: str):
        self.variables = variables

    def fit(self, X:pd.DataFrame, y:pd.Series=None):
        return self

    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        def to_second(x):
            hours = x // 10000
            minutes = (x % 10000) // 100
            seconds = x % 100
            total_seconds = hours * 3600 + minutes * 60 + seconds
            return total_seconds
            
        X = X.copy()
        X[self.variables] = X[self.variables].apply(to_second)
        scaler = MinMaxScaler()
        X[self.variables] = scaler.fit_transform(X[self.variables].values.reshape(-1, 1))

        return X

        
def pipeline(columns: List) -> Tuple[Pipeline, List]:
    
    # transform obj
    obj_columns = [(index, c) for index, c in enumerate(columns) if c in config.log_config.categorical_features]
    transform_object = ColumnTransformer(
        transformers=[
            ('transform_obj', TargetEncoder(target_type='binary', random_state=42), list(zip(*obj_columns))[0])
        ],
        remainder='passthrough'
    )
    obj_columns_names = list(list(zip(*obj_columns))[1])
    new_order_columns = [c for c in columns if c not in obj_columns_names]
    new_order_columns = obj_columns_names + new_order_columns

    # fill all na values
    na_columns = [(index, c) for index, c in enumerate(new_order_columns) if c in config.log_config.vars_with_na]
    fill_na = ColumnTransformer(
        transformers=[
            ('fill_na', SimpleImputer(strategy="most_frequent"), list(zip(*na_columns))[0])
        ],
        remainder='passthrough'
    )

    na_columns_names = list(list(zip(*na_columns))[1])
    new_order_columns = [c for c in new_order_columns if c not in na_columns_names]
    new_order_columns = na_columns_names + new_order_columns

    steps = [
            ('time_transformation', TimeTransformer(variables=config.log_config.time_transform)),
            ('obj_transformation', transform_object),
            ('na_values_imputation', fill_na),
            ('scaler', RobustScaler()),        
        ]

    pipe = Pipeline(steps)
    
    return pipe, new_order_columns
    
def feature_transform(path: str) -> None:
    df = pd.read_csv(path)

    to_drop = config.log_config.to_drop
    target = config.log_config.target

    X = df.drop(to_drop+[target], axis=1)
    y = df[target]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=config.log_config.test_size, stratify=y, random_state=config.log_config.random_state)

    pipe, new_order_columns = pipeline(X_train.columns)
    transformed_df = pipe.fit_transform(X_train, y_train)
    transformed_df = pd.DataFrame(transformed_df, columns=new_order_columns)

    joblib.dump(pipe, 'feature_transoform_pipe.joblib')
    transformed_df.to_csv('transformed_training.csv', index=False)
    y_train.to_csv('label.csv', index=False)
    
if __name__ == '__main__':
    feature_transform('test.csv')

    



