import pandas as pd
from config.core import config
from joblib import load
from xgboost import XGBClassifier


def prediction(model: XGBClassifier, predict_path: str) -> None:
    """
    進行最後結果預測，並將結果儲存為final_prediction.csv

    Parameters:
    - model (XGBClassfier): 訓練完的XGBClassfier
    - predict_path (str): 要進行預測的csv file位置

    Returns:
    無
    
    """
    predict_df = pd.read_csv(predict_path)
    to_drop = config.log_config.to_drop
    predict_data = predict_df.drop(to_drop, axis=1)

    pipe = load('feature_transoform_pipe.joblib')
    predict_data = pipe.transform(predict_data)
    
    predictions = model.predict(predict_data)
    
    results = pd.DataFrame({'txkey': predict_df['txkey'].values, 'pred': predictions})
    results.to_csv('final_prediction.csv', index=False)

    

    
    



