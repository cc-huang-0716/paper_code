import pandas as pd
import torch
import os
from util_feature import lstm_preprocessing
from util_feature import create_lstm_sequences, train_and_extract_pytorch

# 建立資料夾存放模型權重
os.makedirs("saved_lstm_models", exist_ok=True)

# 讀取資料與設定 all_rounds
parq = pd.read_parquet("final_data.parquet")
parq.columns = parq.columns.str.strip().str.replace('％', '%')
exclude_list = ['報酬率%', '報酬率％', '年月日', '年份', '代號', '名稱', 'index']
time_steps = 5
all_rounds = [
    {'name': 'Val_1', 'train': range(2015, 2020), 'test': [2020]},
    {'name': 'Val_2', 'train': range(2016, 2021), 'test': [2021]},
    {'name': 'Val_3', 'train': range(2017, 2022), 'test': [2022]},
    {'name': 'Final_Test', 'train': range(2015, 2023), 'test': [2023, 2024, 2025]}
]

for rd in all_rounds:
    print(f" 開始訓練輪次: {rd['name']}")
    
    # 切分資料
    train_years = sorted(list(rd['train']))
    val_year = train_years[-1]
    pure_train_years = train_years[:-1] 
    parq['年份'] = pd.to_datetime(parq['年月日']).dt.year
    
    raw_train = parq[parq['年份'].isin(pure_train_years)].copy()
    raw_val = parq[parq['年份'].isin([val_year])].copy()
    raw_test = parq[parq['年份'].isin(rd['test'])].copy()
    
    df_train = lstm_preprocessing(X=raw_train)
    df_val = lstm_preprocessing(X=raw_val) 
    df_test = lstm_preprocessing(X=raw_test)

    feature_cols = [c for c in df_train.columns if c not in exclude_list]
    print(f"確認特徵維度: {len(feature_cols)}")
    
    # 轉換 3D 序列
    X_train_3d, _ = create_lstm_sequences(df_train, feature_cols, time_steps)
    X_val_3d, _ = create_lstm_sequences(df_val, feature_cols, time_steps)
    
    # 訓練並取得模型權重
    print("啟動 PyTorch 訓練")
    trained_model = train_and_extract_pytorch(
        X_train=X_train_3d, 
        X_val=X_val_3d, 
        encoding_dim=10,
        epochs=50,
        batch_size=256,
        patience=5,
        return_model_only=True 
    )

    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    model_path = f"{save_dir}/lstm_encoder_{rd['name']}.pth"
    torch.save(trained_model.state_dict(), model_path)
    print(f"模型儲存至: {model_path}")