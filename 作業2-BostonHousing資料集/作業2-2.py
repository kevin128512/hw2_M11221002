# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 13:07:54 2023

@author: User
"""
!pip install scikit-learn xgboost
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import numpy as np

# 載入數據集
df = pd.read_csv('BostonHousing.csv')

# 分割特徵和目標變數
X = df.drop('medv', axis=1).values  # 假設'MEDV'是目標欄位的名稱
y = df['medv'].values

# 定義模型
model = XGBRegressor()

# K-fold交叉驗證
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 存儲每個fold的績效指標
mape_scores = []
rmse_scores = []
r2_scores = []

fold = 1
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mape = mean_absolute_percentage_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2 = r2_score(y_test, predictions)

    mape_scores.append(mape)
    rmse_scores.append(rmse)
    r2_scores.append(r2)

    # 打印每個fold的性能
    print(f"Fold {fold} Performance:")
    print(f"    MAPE: {mape}")
    print(f"    RMSE: {rmse}")
    print(f"    R2: {r2}")
    print("--------------------------------------------------")

    fold += 1

# 計算平均績效
avg_mape = np.mean(mape_scores)
avg_rmse = np.mean(rmse_scores)
avg_r2 = np.mean(r2_scores)

print("Average Performance:")
print(f"    Average MAPE: {avg_mape}")
print(f"    Average RMSE: {avg_rmse}")
print(f"    Average R2: {avg_r2}")

# 特徵重要性統計
feature_importances = np.zeros(X.shape[1])

# 重新進行K-fold交叉驗證以計算特徵重要性
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model.fit(X_train, y_train)
    feature_importances += model.feature_importances_

# 計算平均特徵重要性
feature_importances /= kf.n_splits
print("Feature Importances:", feature_importances)

# 根據特徵重要性進行特徵選擇
sorted_idx = np.argsort(feature_importances)[::-1]
for i in range(1, X.shape[1] + 1):
    selected_features = sorted_idx[:i]
    X_selected = X[:, selected_features]

    # 使用選定的特徵重新進行交叉驗證
    mape_scores_selected = []
    rmse_scores_selected = []
    r2_scores_selected = []

    for train_index, test_index in kf.split(X_selected):
        X_train, X_test = X_selected[train_index], X_selected[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mape_scores_selected.append(mean_absolute_percentage_error(y_test, predictions))
        rmse_scores_selected.append(mean_squared_error(y_test, predictions, squared=False))
        r2_scores_selected.append(r2_score(y_test, predictions))

    print(f"Performance with top {i} features:")
    print(f"    MAPE: {np.mean(mape_scores_selected)}")
    print(f"    RMSE: {np.mean(rmse_scores_selected)}")
    print(f"    R2: {np.mean(r2_scores_selected)}")
    print("--------------------------------------------------")