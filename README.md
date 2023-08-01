# COMP9417 Project - Predict Student Performance from Game Play

## 9417-pro - Group members

|      NAME      | Student ID |
|:--------------:|:----------:|
|  Siyuan Wu   |  z5412156 |
| Chao Xu | z5441640 |
| Benedicta Chun | z5260342 |
| Weizhi Chen | z5430533 |
| Liren Ding |  z5369144 |


## Preparation

### Prior to the execution of the programme, the directory structure must be setup as follow in Colab.
```tree
content
|_ _ drive
|	|_ _ MyDrive
|	|	|_ _ 9417project
|	|	|	|_ _models	
|	|	|	|	|_ _LSTM_models	
|	|	|	|	|_ _SVM_models	
|	|	|	|_ _predictions
|	|	|	|	|_ _LSTM_pred.npy	
|	|	|	|	|_ _SVM_pred.npy	
|	|	|	|_ _report
|	|	|	|_ _test.csv
|	|	|	|_ _train.csv
|	|	|	|_ _train_labels.csv
```

### Download the kaggle dataset from the KaggleDatasetDownload.ipynb
1. Open the KaggleDatasetDownload.ipynb in your colab 
2. Follow the introduction and get your own kaggle.json file
3. Run the KaggleDatasetDownload.ipynb to generate zipped dataset
4. Unzip the dataset

## Program execution
Due to the lengthy duration of the training and feature engineering processes, all intermediate 
results are documented in the Colab's notebook. However, if a complete programme run is 
desired, the following execution order must be followed:

1. Run SVM.ipynb to generate SVM_pred.npy
2. Run LSTM.ipynb to generate LSTM_pred.npy
3. Run xgb_final_with_engine.ipynb to get the final prediction result






