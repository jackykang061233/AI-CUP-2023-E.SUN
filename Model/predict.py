import typing as t

import pandas as pd
from config.core import config
from joblib import load


def prediction(model, predict_path):
    predict_df = pd.read_csv(predict_path)
    to_drop = config.log_config.to_drop
    predict_data = predict_df.drop(to_drop, axis=1)

    pipe = load('feature_transoform_pipe.joblib')
    predict_data = pipe.transform(predict_data)
    
    predictions = model.predict(predict_data)
    
    results = pd.DataFrame({'txkey': predict_df['txkey'].values, 'pred': predictions})

    print(results.head())

    
    



