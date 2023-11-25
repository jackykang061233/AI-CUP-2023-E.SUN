import pandas as pd
from config.core import config
import xgboost as xgb

    
def train(X_path: str, y_path: str) -> xgb.XGBClassifier:
    """
    進行xgboost模型訓練

    Parameters:
    - X_path (str): 經過preprocessing的訓練資料位置(不含label)
    - y_path (str): label的資料位置

    Returns:
    - xgb.XGBClassifier: 訓練完的模型
    
    """
    print('--------START TRAINING--------')
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)
    model = xgb.XGBClassifier(**dict(config.log_config.xgb))
    model.fit(X, y)
                     
    print('--------END TRAINING--------')
    return model

    
    

