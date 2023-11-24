from typing import List
from pydantic import BaseModel, validator
from strictyaml import YAML, load


# App
class AppConfig(BaseModel):
    """
    Application-level config
    """
    package_name: str
    training_data: str
    predict_path: str

# Model config
class XgbConfig(BaseModel):
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

class LogConfig(BaseModel):
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
        return [] if value == [''] else value


class Config(BaseModel):
    """Master config object."""
    app_config: AppConfig
    log_config: LogConfig


def get_config() -> Config:
    """Run validation on config values."""
    with open('config/config.yml', "r") as conf_file:
        parsed_config = load(conf_file.read())

    # specify the data attribute from the strictyaml YAML type.
    config = Config(
        app_config=AppConfig(**parsed_config['app_config'].data),
        log_config=LogConfig(**parsed_config['log_config'].data),
    )

    return config

config = get_config()