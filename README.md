# AI-CUP-2023-E.SUN
# 檔案用途:

* Preprocess/: 
  * pipeline.py: 使用sklearn將原始資料轉換
* Model/: 裡面有兩個檔案
  * train.py: load preprocess後的資料進行訓練
  * predict.py: 使用train.py訓練過的模型進行預測
* config/: 裡面有兩個檔案
  * config.yml: 存放訓練所需的configuration
  * core.py: 讀取config.yml並使用pydantic驗證資料的型態
* requirements.txt: 需要的套件
* main.py: 執行整個訓練流程以及預測的部分

# 執行流程:

```
# 安裝所需套件
$ pip install -r requirements.txt 

# 執行資料前處理
$ python Preprocess/pipeline.py 

# training and inference
$ python main.py
```
