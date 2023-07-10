import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb

sns.set()
from sklearn.model_selection import train_test_split

# linear regression
dtypes = {
    'elapsed_time': np.int32,
    'event_name': 'category',
    'name': 'category',
    'level': np.uint8,
    'room_coor_x': np.float32,
    'room_coor_y': np.float32,
    'screen_coor_x': np.float32,
    'screen_coor_y': np.float32,
    'hover_duration': np.float32,
    'text': 'category',
    'fqid': 'category',
    'room_fqid': 'category',
    'text_fqid': 'category',
    'fullscreen': 'category',
    'hq': 'category',
    'music': 'category',
    'level_group': 'category'}

data1 = pd.read_csv('C:\\Users\\dlr\\Desktop\\Assi\\comp9417\\test.csv', dtype=dtypes)

# print("Full train dataset shape is {}".format(data1.shape))

# data1.info()
# print(data1.columns)
# print(data1.describe())
def changeNulltozero(data1):
    data1['page'] = data1['page'].astype('object')
    data1['page'] = data1['page'].fillna(0, axis=0)
    data1['page'] = data1['page'].astype('category')
    data1['room_coor_x'] = data1['room_coor_x'].fillna(0, axis=0)
    data1['room_coor_y'] = data1['room_coor_y'].fillna(0, axis=0)
    data1['screen_coor_x'] = data1['screen_coor_x'].fillna(0, axis=0)
    data1['screen_coor_y'] = data1['screen_coor_y'].fillna(0, axis=0)
    data1['hover_duration'] = data1['hover_duration'].fillna(0, axis=0)
    data1['text'] = data1['text'].astype('object')
    data1['text'] = data1['text'].fillna(0, axis=0)
    data1['text'] = data1['text'].astype('category')
    data1['fqid'] = data1['fqid'].astype('object')
    data1['fqid'] = data1['fqid'].fillna(0, axis=0)
    data1['fqid'] = data1['fqid'].astype('category')
    data1['text_fqid'] = data1['text_fqid'].astype('object')
    data1['text_fqid'] = data1['text_fqid'].fillna(0, axis=0)
    data1['text_fqid'] = data1['text_fqid'].astype('category')
    return data1


data1 = changeNulltozero(data1)

#print(data1.isna().sum())
#print(data2.isna().sum())
# find the highest positive relative between two var
correlacao_notas = data1.corr(numeric_only=True)
plt.figure(figsize=(10, 6))
sns.heatmap(correlacao_notas, annot=True, cmap="BrBG", vmin=-1, vmax=1)
plt.xticks(rotation=45)
plt.show()

# Split the data into train and test sets
Train = data1.select_dtypes(include=['float64', 'int64'])
X = Train.drop(columns=['session_id', 'session_level'])
y = Train['session_level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Train a Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict probabilities using the Random Forest model
rf_pred_proba = rf_model.predict_proba(X_test)
rf_pred_score = rf_pred_proba[:, 1].mean()  # Calculate the mean of the positive class probabilities

# Train an XGBoost classifier with the calculated base_score
gbm_model = xgb.XGBClassifier(base_score=rf_pred_score, booster='gbtree', colsample_bylevel=1,
                              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
                              importance_type='gain', interaction_constraints='',
                              learning_rate=0.01, max_delta_step=0, max_depth=6,
                              min_child_weight=1, monotone_constraints='()',
                              n_estimators=100, n_jobs=4, num_parallel_tree=1,
                              random_state=42, reg_alpha=0, reg_lambda=1,
                              subsample=0.8, tree_method='exact', validate_parameters=1,
                              verbosity=None)

# Train the XGBoost model
gbm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_proba = gbm_model.predict_proba(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)

# Calculate accuracy and AUC
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
print('Accuracy:', accuracy)
print('AUC:', auc)