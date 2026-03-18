import pandas as pd
import os
import numpy as np
import time
from sklearn.decomposition import PCA
from util_feature import pca_preprocessing, lstm_preprocessing, create_lstm_sequences, load_and_extract_features, save_algorithm_results
from util_algorithms import run_rf_strategy 

# 讀檔
start = time.time()
parq = pd.read_parquet("final_data.parquet")

parq.columns = parq.columns.str.strip().str.replace('％', '%')
parq['年份'] = pd.to_datetime(parq['年月日']).dt.year
print(f"數據載入完成，形狀: {parq.shape}，耗時 {time.time() - start:.2f} 秒")

exclude_list = ['報酬率%', '報酬率％', '年月日', '年份', '代號', '名稱', 'index']
numeric_cols = parq.select_dtypes(include=['number']).columns
feat_cols = [c for c in parq.columns if c not in exclude_list and c in numeric_cols]
n_features = len(feat_cols)
print(f"確認模型特徵維度: {n_features} 維")

# 滾動視窗設定
all_rounds = [
    {'name': 'Val_1', 'train': range(2015, 2020), 'test': [2020]},
    {'name': 'Val_2', 'train': range(2016, 2021), 'test': [2021]},
    {'name': 'Val_3', 'train': range(2017, 2022), 'test': [2022]},
    {'name': 'Final_Test', 'train': range(2015, 2023), 'test': [2023, 2024, 2025]}
]

pca_rf_results = []
lstm_rf_results = []

for rd in all_rounds:
    print(f"執行輪次: {rd['name']}")
    
# pca
    # 數據切割與降維
    train_parq = parq[parq['年份'].isin(rd['train'])].sort_values(['代號', '年月日']).reset_index(drop=True)
    test_parq = parq[parq['年份'].isin(rd['test'])].sort_values(['代號', '年月日']).reset_index(drop=True)
    
    train_parq['target_ret'] = train_parq.groupby('代號')['報酬率%'].shift(-1)
    test_parq['target_ret'] = test_parq.groupby('代號')['報酬率%'].shift(-1)

    df_pca_train = pca_preprocessing(train_parq)
    df_pca_test = pca_preprocessing(test_parq)
    
    # 填補缺失值並對齊答案
    df_pca_train['報酬率%'] = train_parq['報酬率%'].ffill().fillna(0)
    df_pca_test['報酬率%'] = test_parq['報酬率%'].ffill().fillna(0)
    df_pca_test['年月日'] = test_parq['年月日']

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

# LSTM
    # 數據切割與萃取
    train_years = sorted(list(rd['train']))
    val_year = train_years[-1]
    pure_train_years = train_years[:-1]
    
    raw_train = parq[parq['年份'].isin(pure_train_years)].copy()
    raw_val = parq[parq['年份'].isin([val_year])].copy()
    raw_test = parq[parq['年份'].isin(rd['test'])].copy()
    
    df_lstm_train = lstm_preprocessing(raw_train)
    df_lstm_val = lstm_preprocessing(raw_val)
    df_lstm_test = lstm_preprocessing(raw_test)
    
    # 轉換 3D 序列
    time_steps = 5
    X_train_3d, idx_train_feat, idx_train_y = create_lstm_sequences(df_lstm_train, feat_cols, time_steps, predict_horizon=1)
    X_val_3d, idx_val_feat, idx_val_y = create_lstm_sequences(df_lstm_val, feat_cols, time_steps, predict_horizon=1)
    X_test_3d, idx_test_feat, idx_test_y = create_lstm_sequences(df_lstm_test, feat_cols, time_steps, predict_horizon=1)
    
    # 載入模型
    model_path = f"saved_models/lstm_encoder_{rd['name']}.pth"
    print(f"載入模型並萃取特徵: {model_path}")
    
    train_enc = load_and_extract_features(X_train_3d, time_steps, n_features, model_path)
    val_enc = load_and_extract_features(X_val_3d, time_steps, n_features, model_path)
    test_enc = load_and_extract_features(X_test_3d, time_steps, n_features, model_path)
    
    # 特徵重建與對齊
    lstm_cols = [f'LSTM_F{i+1}' for i in range(10)]
    
    # 確保長度對齊
    df_lstm_train_final = pd.DataFrame(train_enc, columns=lstm_cols)
    df_lstm_train_final['報酬率%'] = raw_train.loc[idx_train_y, '報酬率%'].values

    df_lstm_test_final = pd.DataFrame(test_enc, columns=lstm_cols)
    df_lstm_test_final['報酬率%'] = raw_test.loc[idx_test_y, '報酬率%'].values
    df_lstm_test_final['年月日'] = raw_test.loc[idx_test_y, '年月日'].values
    df_lstm_test_final['代號'] = raw_test.loc[idx_test_y, '代號'].values


    print(f"執行 Random Forest 策略回測")
    
    res_pca = run_rf_strategy(df_pca_train_final, df_pca_test_final, pc_cols, algo_name="PCA_RandomForest")
    pca_rf_results.append(res_pca)
    
    res_lstm = run_rf_strategy(df_lstm_train_final, df_lstm_test_final, lstm_cols, algo_name="LSTM_RandomForest")
    lstm_rf_results.append(res_lstm)

# 儲存與產出結果
save_algorithm_results(pca_rf_results, algo_name="RandomForest_Baseline_PCA")
save_algorithm_results(lstm_rf_results, algo_name="RandomForest_Experimental_LSTM")

os.makedirs("thesis_results/predictions", exist_ok=True)
rf_pca_preds = pd.concat([res['test_data'] for res in pca_rf_results])
rf_lstm_preds = pd.concat([res['test_data'] for res in lstm_rf_results])

rf_pca_preds[['年月日', '代號', '報酬率%', 'y_pred']].to_parquet("thesis_results/predictions/PCA_rf_preds.parquet")
rf_lstm_preds[['年月日', '代號', '報酬率%', 'y_pred']].to_parquet("thesis_results/predictions/LSTM_rf_preds.parquet")

print("任務完成")