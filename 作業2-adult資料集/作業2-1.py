# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 12:15:25 2023

@author: User
"""
!pip install scikit-learn xgboost
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import time

# 載入資料集
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
train_data = pd.read_csv('adult.data', names=column_names)
test_data = pd.read_csv('adult.test', names=column_names, skiprows=1) # 跳過第一行，因為它是不規則的

# 將類別資料轉換為數字
train_data = pd.get_dummies(train_data)
test_data = pd.get_dummies(test_data)

# 確保訓練集和測試集有相同的欄位
common_columns = train_data.columns.intersection(test_data.columns)
train_data = train_data[common_columns]
test_data = test_data[common_columns]

# 分離特徵和標籤
X_train = train_data.drop('hours-per-week', axis=1)
y_train = train_data['hours-per-week']
X_test = test_data.drop('hours-per-week', axis=1)
y_test = test_data['hours-per-week']

# 特徵標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 定義模型
models = {
    'KNN': KNeighborsRegressor(),
    'SVR': SVR(),
    'RandomForest': RandomForestRegressor(),
    'XGBoost': XGBRegressor()
}

# 模型訓練及預測
for name, model in models.items():
    start_time = time.time()
    
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    
    end_time = time.time()
    
    mape = mean_absolute_percentage_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2 = r2_score(y_test, predictions)
    
    # 打印每個模型的名稱和性能指標
    print(f'{name}演算法')
    print(f'    MAPE: {mape}')
    print(f'    RMSE: {rmse}')
    print(f'    R2: {r2}')
    print(f'    Time taken: {end_time - start_time} seconds')
    print('--------------------------------------------------')



