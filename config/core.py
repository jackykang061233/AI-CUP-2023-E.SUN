"""
印證config.yml裡的configuration型態
"""
from typing import List
from pydantic import BaseModel, validator
from strictyaml import YAML, load


# xgboost config
class XgbConfig(BaseModel):
    """ 定義xgboost parameters 的型態 """
    objective: str
    random_state: int
    scale_pos_weight: float
    n_estimators: int
    learning_rate: float
    device: str
    gamma: float
    reg_alpha: float
    reg_lambda: float
    max_depth: int
    min_child_weight: int
    colsample_bytree: float
    subsample: float
    n_jobs: int

# Model config
class LogConfig(BaseModel):
    """ 定義整個模型 parameters 的型態 """
    target: str
    used_model: str
    to_drop: List[str]
    object_features: List[str]
    numeric_features: List[str]
    categorical_features: List[str]
    random_state: int
    test_size: float
    vars_with_na: List[str]
    time_transform: str
    xgb: XgbConfig

    @validator("to_drop",pre=True, allow_reuse=True)
    def convert_empty_string_to_none(cls, value):
        """ 如果沒有任何 to_drop的話就return空的list，不然return原值 """
        return [] if value == [''] else value

class Config(BaseModel):
    """ 統整所有config的物件 """
    log_config: LogConfig
    

def get_config() -> Config:
    """ load預先設定的configuration後將它轉換成Config物件 """
    with open('config/config.yml', "r") as conf_file:
        parsed_config = load(conf_file.read())

    config = Config(
        log_config=LogConfig(**parsed_config['log_config'].data),
    )

    return config

config = get_config()
