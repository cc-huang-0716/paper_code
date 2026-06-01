import os
import time
import pandas as pd
import torch

from util_feature import lstm_preprocessing, create_lstm_sequences, train_and_extract_pytorch

DATA_PATH = "final_data.parquet"
SAVE_DIR = "saved_models"
TIME_STEPS = 5
ENCODING_DIM = 10
EPOCHS = 50
BATCH_SIZE = 256
PATIENCE = 5
EXCLUDE_LIST = ["報酬率%", "報酬率％", "年月日", "年份", "代號", "名稱", "index", "target_ret"]


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    start = time.time()
    parq = pd.read_parquet(DATA_PATH)
    parq.columns = parq.columns.str.strip().str.replace("％", "%")
    parq["年月日"] = pd.to_datetime(parq["年月日"])
    parq["年份"] = parq["年月日"].dt.year
    print(f"資料載入完成 {parq.shape}，耗時 {time.time() - start:.2f} 秒")

    all_rounds = [
        {"name": "Val_1", "train": range(2015, 2020), "test": [2020]},
        {"name": "Val_2", "train": range(2016, 2021), "test": [2021]},
        {"name": "Val_3", "train": range(2017, 2022), "test": [2022]},
        {"name": "Final_Test", "train": range(2015, 2023), "test": [2023, 2024, 2025]},
    ]

    for rd in all_rounds:
        print(f"開始訓練輪次: {rd['name']}")
        train_years = sorted(list(rd["train"]))
        val_year = train_years[-1]
        pure_train_years = train_years[:-1]

        raw_train = parq[parq["年份"].isin(pure_train_years)].copy()
        raw_val = parq[parq["年份"].isin([val_year])].copy()

        df_train = lstm_preprocessing(raw_train)
        df_val = lstm_preprocessing(raw_val)
        feature_cols = [c for c in df_train.select_dtypes(include=["number"]).columns if c not in EXCLUDE_LIST]
        print(f"確認特徵維度: {len(feature_cols)}")

        X_train_3d, _, _ = create_lstm_sequences(df_train, feature_cols, TIME_STEPS, predict_horizon=1)
        X_val_3d, _, _ = create_lstm_sequences(df_val, feature_cols, TIME_STEPS, predict_horizon=1)
        print(f"X_train: {X_train_3d.shape}, X_val: {X_val_3d.shape}")

        trained_model = train_and_extract_pytorch(
            X_train=X_train_3d,
            X_val=X_val_3d,
            round_name=rd["name"],
            encoding_dim=ENCODING_DIM,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            patience=PATIENCE,
            return_model_only=True,
        )

        model_path = f"{SAVE_DIR}/lstm_encoder_{rd['name']}.pth"
        torch.save(trained_model.state_dict(), model_path)
        print(f"模型儲存至: {model_path}")


if __name__ == "__main__":
    main()
