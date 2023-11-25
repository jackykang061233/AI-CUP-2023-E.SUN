from Model.train import train
from Model.predict import prediction
from Preprocess.pipeline import TimeTransformer 


if __name__ == '__main__':
    model = train('transformed_training.csv', 'label.csv') # 模型訓練
    prediction(model, 'predict.csv') # 模型預測