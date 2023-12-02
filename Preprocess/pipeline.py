"""
進行preprocessing並
- 將input X儲存為transformed_training.csv
- 將input y儲存為label.csv
- 將使用過的pipeline儲存為feature_transoform_pipe.joblib，以便之後在prediction時使用

input:
- training.py: 訓練資料位置
"""
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, TargetEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import joblib
from typing import Tuple, List, Union

import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from config.core import config

class TimeTransformer(BaseEstimator, TransformerMixin):
    """
    將要進行轉換的features欄位轉換成秒並使用MinMaxScaler進行normalization

    Parameters:
    - variables (str or list of str): 要進行轉換的features

    Methods:
    - fit(X, y=None): Fit the transformer.
    - transform(X): 將feature轉換成秒後使用MinMaxScaler進行normalization

    """
    
    def __init__(self, variables: Union[str, List[str]]):
        """
        TimeTransformer class的Constructor

        Parameters:
        - variables (str or list of str): 要進行轉換的features
        """
        self.variables = variables

    def fit(self, X:pd.DataFrame, y:pd.Series=None):
        """
        Fit the transformer

        Parameters:
        - X (pd.DataFrame): input的DataFrame
        - y (pd.Series): input的target (可以省略)

        Returns:
        TimeTransformer: fit後的TimeTransformer instance
        """
        return self

    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        將feature轉換成秒後使用MinMaxScaler進行normalization

        Parameters:
        - X (pd.DataFrame): input的DataFrame

        Returns:
        pd.DataFrame: 轉換後的DataFrame.
        """
        def to_second(x: int):
            """
            將原數轉換成秒

            Parameters:
            - X (int): input的值 從 000000(在這裡為0) 到 235959 前兩位為小時中間兩位為分後兩位為秒

            Returns:
            int: 轉換後的秒數
            """
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
    
class NewNAColumn(BaseEstimator, TransformerMixin):
    """ create new column for na value which indicates if it is na 1 for nan value 0 for not nan value"""
    def __init__(self, col):
        self.col = col
        
    def fit(self, X=None, y=None):
        return self
    
    def transform(self, X, y=None):
        for col in self.col:
            X[col+'_na'] = np.where(X[col].isnull(), 1, 0)
            X[col] = X[col].fillna(X[col].nunique())
        return X

        
def pipeline(columns: List) -> Tuple[Pipeline, List]:
    """
    將原始資料進行feature engineering然後
    - 將X儲存為transformed_training.csv
    - 將y儲存為label.csv
    - 並將使用過的pipeline儲存為feature_transoform_pipe.joblib，以便之後在prediction時使用

    Parameters:
    - path (str): 原始訓練資料位置

    Returns:
    - pipe (Pipeline): 融合所有feature engineering steps的pipeline
    - new_order_columns (list): 因為經過columnstransformer後feature的相對會被打亂所以需要一個更新過後的feature排列位置
    
    """
    
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
            ('add_na_column', NewNAColumn(col=config.log_config.add_na_column)),
            ('obj_transformation', transform_object),
            ('na_values_imputation', fill_na),
            ('scaler', RobustScaler()),        
        ]

    pipe = Pipeline(steps)
    
    return pipe, new_order_columns
    
def feature_transform(path: str) -> None:
    """
    將原始資料進行feature engineering然後
    - 將X儲存為transformed_training.csv
    - 將y儲存為label.csv
    - 並將使用過的pipeline儲存為feature_transoform_pipe.joblib，以便之後在prediction時使用

    Parameters:
    - path (str): 原始訓練資料位置

    Returns:
    無
    
    """
    df = pd.read_csv(path)

    to_drop = config.log_config.to_drop
    target = config.log_config.target

    X = df.drop(to_drop+[target], axis=1)
    y = df[target]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=config.log_config.test_size, stratify=y, random_state=config.log_config.random_state)

    pipe, new_order_columns = pipeline(X_train.columns)
    transformed_df = pipe.fit_transform(X_train, y_train)
    
    na_column_index = []
    for i, col in enumerate(new_order_columns):
        if col in config.log_config.add_na_column:
            na_column_index.append((i, 'col_na'))
    for i in range(len(na_column_index)):
        new_order_columns.insert(na_column_index[i][0]+i+1, na_column_index[i][1])
    
    transformed_df = pd.DataFrame(transformed_df, columns=new_order_columns)

    joblib.dump(pipe, 'feature_transoform_pipe.joblib')
    transformed_df.to_csv('transformed_training.csv', index=False)
    y_train.to_csv('label.csv', index=False)
    
if __name__ == '__main__':
    feature_transform('training.csv')

    