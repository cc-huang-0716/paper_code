import pandas as pd
import os
from pathlib import Path
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dataset import data_processing
from util_feature import pca_preprocessing
from util_algorithms import multi_reg
from util_feature import save_algorithm_results
from util_feature import pca_result
from util_feature import plot_pca_loadings_heatmap
from util_feature import load_and_extract_features
from util_feature import create_lstm_sequences
from util_feature import lstm_preprocessing

# 載入數據 
start = time.time()
parq = pd.read_parquet("final_data.parquet")
#parq = data_processing()
num_df = parq.select_dtypes(include=["number"]).dropna()
parq.columns = parq.columns.str.strip().str.replace('％', '%')
print("讀檔完成", parq.shape, "耗時", time.time() - start,"秒")
exclude_list = ['報酬率%', '報酬率％', '年月日', '年份', '代號', '名稱', 'index']
numeric_cols = parq.select_dtypes(include=['number']).columns
feat_cols = [c for c in parq.columns if c not in exclude_list and c in numeric_cols]

# 線性假設下的預測器
    # 訓練，驗證與測試
        # 訓練期和測試期滾動調整
start = time.time()
all_rounds = [
    {'name': 'Val_1', 'train': range(2015, 2020), 'test': [2020]},
    {'name': 'Val_2', 'train': range(2016, 2021), 'test': [2021]},
    {'name': 'Val_3', 'train': range(2017, 2022), 'test': [2022]},
    {'name': 'Final_Test', 'train': range(2015, 2023), 'test': [2023, 2024, 2025]}
]

reg_predictions_pca = [] 

for rd in all_rounds:
    print(f" 正在執行: {rd['name']} (訓練: {list(rd['train'])})")
    current_round_name = rd['name']

    # 切割數據
    X_cs = pca_preprocessing(X=parq)
    parq['年份'] = pd.to_datetime(parq['年月日']).dt.year

    train_parq = parq[parq['年份'].isin(rd['train'])].sort_values(['代號', '年月日']).reset_index(drop=True).copy()
    test_parq = parq[parq['年份'].isin(rd['test'])].sort_values(['代號', '年月日']).reset_index(drop=True).copy()
    train_parq['target_ret'] = train_parq.groupby('代號')['報酬率%'].shift(-1)
    test_parq['target_ret'] = test_parq.groupby('代號')['報酬率%'].shift(-1)
    #print(f"train_parq 目前有的欄位: {train_parq.columns.tolist()}")

    df_pca_train = pca_preprocessing(train_parq)
    df_pca_test = pca_preprocessing(test_parq)

    pca = PCA(n_components=10)
    train_pca_v = pca.fit_transform(df_pca_train[feat_cols])
    test_pca_v = pca.transform(df_pca_test[feat_cols])

    pc_cols = [f'PC{i+1}' for i in range(10)]

    df_pca_train_final = pd.DataFrame(train_pca_v, columns=pc_cols, index=train_parq.index)
    df_pca_train_final['報酬率%'] = train_parq['target_ret'].values

    df_pca_test_final = pd.DataFrame(test_pca_v, columns=pc_cols, index=test_parq.index)
    df_pca_test_final['報酬率%'] = test_parq['target_ret'].values
    df_pca_test_final['年月日'] = test_parq['年月日'].values
    df_pca_test_final['代號'] = test_parq['代號'].values

    train_mask = df_pca_train_final['報酬率%'].notna()
    test_mask = df_pca_test_final['報酬率%'].notna()

    df_pca_train_final = df_pca_train_final.loc[train_mask].reset_index(drop=True)
    df_pca_test_final = df_pca_test_final.loc[test_mask].reset_index(drop=True)
    
    # 模型部署，樣本外預測，指標計算 
    reg_result = multi_reg(df_pca_train_final, df_pca_test_final, features=pc_cols, y_col="報酬率%", date_col="年月日", annualization=252)
    reg_predictions_pca.append(reg_result)

# OLS 的結果
save_algorithm_results(reg_predictions_pca, algo_name="OLS_Baseline")
print("檢定完成", num_df.shape, "耗時", time.time() - start,"秒")

# 整合所有預測結果
#final_df = pd.concat(all_predictions)
# 整合績效報表
#metrics_df = pd.DataFrame(all_predictions)




# 非線性假設的預測器
    # 訓練，驗證與測試
        # 訓練期和測試期滾動調整

reg_predictions_lstm = []
time_steps = 5

for rd in all_rounds:

    print(f"目前執行輪次: {rd['name']} ")
    
    # 切分資料與預處理
    train_years = sorted(list(rd['train']))
    val_year = train_years[-1]
    pure_train_years = train_years[:-1] 
    
    raw_train = parq[parq['年份'].isin(pure_train_years)].copy()
    raw_val = parq[parq['年份'].isin([val_year])].copy()
    raw_test = parq[parq['年份'].isin(rd['test'])].copy()

    df_lstm_train = lstm_preprocessing(raw_train)
    df_lstm_val = lstm_preprocessing(raw_val)
    df_lstm_test = lstm_preprocessing(raw_test)

    exclude_cols = ['報酬率%', '報酬率％', '年月日', '年份', '代號', '名稱', 'index']
    feature_cols = [c for c in df_lstm_train.columns if c not in exclude_cols]
    n_features = len(feature_cols)

    print(f"檢查特徵維度: {n_features} 維")

    # 轉換 3D 序列
    X_train_3d, idx_train_feat, idx_train_y = create_lstm_sequences(
        raw_train, feat_cols, time_steps, predict_horizon=1
    )
    X_val_3d, idx_val_feat, idx_val_y = create_lstm_sequences(
        raw_val, feat_cols, time_steps, predict_horizon=1
    )
    X_test_3d, idx_test_feat, idx_test_y = create_lstm_sequences(
        raw_test, feat_cols, time_steps, predict_horizon=1
    )

    # 直接讀取預先訓練好的 LSTM 模型並萃取特徵
    model_path = f"saved_models/lstm_encoder_{rd['name']}.pth"
    print(f"載入模型: {model_path}")

    train_encoded = load_and_extract_features(X_train_3d, time_steps, n_features, model_path)
    val_encoded = load_and_extract_features(X_val_3d, time_steps, n_features, model_path)
    test_encoded = load_and_extract_features(X_test_3d, time_steps, n_features, model_path)
    
    # 特徵重建與資料對齊
    lstm_cols = [f'LSTM_F{i+1}' for i in range(10)]
    
    df_train_final = pd.DataFrame(train_encoded, columns=lstm_cols)
    df_train_final['報酬率%'] = raw_train.loc[idx_train_y, '報酬率%'].values
    
    df_val_final = pd.DataFrame(val_encoded, columns=lstm_cols)
    df_val_final['報酬率%'] = raw_val.loc[idx_val_y, '報酬率%'].values
    
    df_reg_train = pd.concat([df_train_final, df_val_final])
    
    df_test_final = pd.DataFrame(test_encoded, columns=lstm_cols)
    df_test_final['報酬率%'] = raw_test.loc[idx_test_y, '報酬率%'].values
    df_test_final['年月日'] = raw_test.loc[idx_test_y, '年月日'].values
    df_test_final['代號'] = raw_test.loc[idx_test_y, '代號'].values

    df_reg_train = df_train_final.copy()
    
    # 多元迴歸預測與多空投資組合建構
    print("OLS 策略回測")
    reg_result = multi_reg(
        train_df=df_reg_train, 
        test_df=df_test_final, 
        features=lstm_cols, 
        y_col="報酬率%", 
        date_col="年月日", 
        annualization=252
    )
    reg_predictions_lstm.append(reg_result)

# 匯出結果
save_algorithm_results(reg_predictions_lstm, algo_name="LSTM_OLS_Strategy")

os.makedirs("thesis_results/predictions", exist_ok=True)
df_pca_preds = pd.concat([res['test_data'] for res in reg_predictions_pca])
df_lstm_preds = pd.concat([res['test_data'] for res in reg_predictions_lstm])

df_pca_preds[['年月日', '代號', '報酬率%', 'y_pred']].to_parquet("thesis_results/predictions/PCA_OLS_preds.parquet")
df_lstm_preds[['年月日', '代號', '報酬率%', 'y_pred']].to_parquet("thesis_results/predictions/LSTM_OLS_preds.parquet")
print("結束，迴歸兩個比較結果出爐")
