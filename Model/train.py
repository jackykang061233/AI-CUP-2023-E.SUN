# basic packags
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import sys
from config.core import config
import xgboost as xgb

    
def train(X_path: str, y_path: str):
    print('--------START TRAINING--------')
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)
    model = xgb.XGBClassifier(**dict(config.log_config.xgb))
    model.fit(X, y)
                     
    print('--------END TRAINING--------')
    return model

    
    

