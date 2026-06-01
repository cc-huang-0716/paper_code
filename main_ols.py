import os
import time
import pandas as pd
from sklearn.decomposition import PCA

from util_feature import (
    pca_preprocessing,
    pca_result,
    plot_pca_loadings_heatmap,
    lstm_preprocessing,
    create_lstm_sequences,
    load_and_extract_features,
    save_algorithm_results,
)
from util_algorithms import multi_reg

SAVE_ONLY_FINAL = False
DATA_PATH = "final_data.parquet"
PRED_DIR = "thesis_results/predictions"
TIME_STEPS = 5
N_COMPONENTS = 10
EXCLUDE_LIST = ["報酬率%", "報酬率％", "年月日", "年份", "代號", "名稱", "index", "target_ret"]


def get_feature_cols(df):
    numeric_cols = df.select_dtypes(include=["number"]).columns
    return [c for c in numeric_cols if c not in EXCLUDE_LIST]


def build_pca_features(train_parq, test_parq, feat_cols, rd_name):
    train_parq = train_parq.copy()
    test_parq = test_parq.copy()
    train_parq["target_ret"] = train_parq.groupby("代號")["報酬率%"].shift(-1)
    test_parq["target_ret"] = test_parq.groupby("代號")["報酬率%"].shift(-1)

    df_pca_train = pca_preprocessing(train_parq)
    df_pca_test = pca_preprocessing(test_parq)

    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    train_pca_v = pca.fit_transform(df_pca_train[feat_cols])
    test_pca_v = pca.transform(df_pca_test[feat_cols])
    pc_cols = [f"PC{i + 1}" for i in range(N_COMPONENTS)]

    # 只在 Final_Test 輸出 PCA 圖表，避免驗證輪次產太多圖。
    if rd_name == "Final_Test":
        loadings = pca_result(pca, feat_cols, round_name=rd_name)
        plot_pca_loadings_heatmap(loadings, round_name=rd_name)

    df_train_final = pd.DataFrame(train_pca_v, columns=pc_cols, index=train_parq.index)
    df_train_final["報酬率%"] = train_parq["target_ret"].values

    df_test_final = pd.DataFrame(test_pca_v, columns=pc_cols, index=test_parq.index)
    df_test_final["報酬率%"] = test_parq["target_ret"].values
    df_test_final["年月日"] = test_parq["年月日"].values
    df_test_final["代號"] = test_parq["代號"].values

    df_train_final = df_train_final.loc[df_train_final["報酬率%"].notna()].reset_index(drop=True)
    df_test_final = df_test_final.loc[df_test_final["報酬率%"].notna()].reset_index(drop=True)
    return df_train_final, df_test_final, pc_cols


def build_lstm_features(parq, rd, feat_cols):
    train_years = sorted(list(rd["train"]))
    val_year = train_years[-1]
    pure_train_years = train_years[:-1]

    raw_train = parq[parq["年份"].isin(pure_train_years)].copy()
    raw_val = parq[parq["年份"].isin([val_year])].copy()
    raw_test = parq[parq["年份"].isin(rd["test"])].copy()

    df_lstm_train = lstm_preprocessing(raw_train)
    df_lstm_val = lstm_preprocessing(raw_val)
    df_lstm_test = lstm_preprocessing(raw_test)

    lstm_feat_cols = [c for c in feat_cols if c in df_lstm_train.columns]
    X_train_3d, idx_train_feat, idx_train_y = create_lstm_sequences(df_lstm_train, lstm_feat_cols, TIME_STEPS, predict_horizon=1)
    X_val_3d, idx_val_feat, idx_val_y = create_lstm_sequences(df_lstm_val, lstm_feat_cols, TIME_STEPS, predict_horizon=1)
    X_test_3d, idx_test_feat, idx_test_y = create_lstm_sequences(df_lstm_test, lstm_feat_cols, TIME_STEPS, predict_horizon=1)

    model_path = f"saved_models/lstm_encoder_{rd['name']}.pth"
    n_features = len(lstm_feat_cols)
    train_enc = load_and_extract_features(X_train_3d, TIME_STEPS, n_features, model_path)
    val_enc = load_and_extract_features(X_val_3d, TIME_STEPS, n_features, model_path)
    test_enc = load_and_extract_features(X_test_3d, TIME_STEPS, n_features, model_path)

    lstm_cols = [f"LSTM_F{i + 1}" for i in range(train_enc.shape[1])]
    df_train = pd.DataFrame(train_enc, columns=lstm_cols)
    df_train["報酬率%"] = df_lstm_train.loc[idx_train_y, "報酬率%"].values

    df_val = pd.DataFrame(val_enc, columns=lstm_cols)
    df_val["報酬率%"] = df_lstm_val.loc[idx_val_y, "報酬率%"].values

    # OLS 預測器可把 validation 也放回訓練集，和 PCA 訓練期間一致。
    df_reg_train = pd.concat([df_train, df_val], ignore_index=True)

    df_test = pd.DataFrame(test_enc, columns=lstm_cols)
    df_test["報酬率%"] = df_lstm_test.loc[idx_test_y, "報酬率%"].values
    df_test["年月日"] = df_lstm_test.loc[idx_test_y, "年月日"].values
    df_test["代號"] = df_lstm_test.loc[idx_test_y, "代號"].values
    df_test = df_test.loc[df_test["報酬率%"].notna()].reset_index(drop=True)
    return df_reg_train, df_test, lstm_cols


def main():
    start = time.time()
    parq = pd.read_parquet(DATA_PATH)
    parq.columns = parq.columns.str.strip().str.replace("％", "%")
    parq["年月日"] = pd.to_datetime(parq["年月日"])
    parq["年份"] = parq["年月日"].dt.year
    feat_cols = get_feature_cols(parq)
    print(f"讀檔完成 {parq.shape}，特徵數={len(feat_cols)}，耗時 {time.time() - start:.2f} 秒")

    all_rounds = [
        {"name": "Val_1", "train": range(2015, 2020), "test": [2020]},
        {"name": "Val_2", "train": range(2016, 2021), "test": [2021]},
        {"name": "Val_3", "train": range(2017, 2022), "test": [2022]},
        {"name": "Final_Test", "train": range(2015, 2023), "test": [2023, 2024, 2025]},
    ]

    reg_predictions_pca = []
    reg_predictions_lstm = []

    for rd in all_rounds:
        print(f"正在執行 {rd['name']}：train={list(rd['train'])}, test={rd['test']}")
        train_parq = parq[parq["年份"].isin(rd["train"])].sort_values(["代號", "年月日"]).reset_index(drop=True)
        test_parq = parq[parq["年份"].isin(rd["test"])].sort_values(["代號", "年月日"]).reset_index(drop=True)

        df_pca_train, df_pca_test, pc_cols = build_pca_features(train_parq, test_parq, feat_cols, rd["name"])
        df_lstm_train, df_lstm_test, lstm_cols = build_lstm_features(parq, rd, feat_cols)

        res_pca = multi_reg(df_pca_train, df_pca_test, features=pc_cols, y_col="報酬率%", date_col="年月日", annualization=252)
        res_pca["metrics"]["algo"] = "PCA_OLS"
        res_lstm = multi_reg(df_lstm_train, df_lstm_test, features=lstm_cols, y_col="報酬率%", date_col="年月日", annualization=252)
        res_lstm["metrics"]["algo"] = "LSTM_OLS"

        if (not SAVE_ONLY_FINAL) or rd["name"] == "Final_Test":
            reg_predictions_pca.append(res_pca)
            reg_predictions_lstm.append(res_lstm)

    save_algorithm_results(reg_predictions_pca, algo_name="OLS_Baseline_PCA")
    save_algorithm_results(reg_predictions_lstm, algo_name="OLS_Experimental_LSTM")

    os.makedirs(PRED_DIR, exist_ok=True)
    df_pca_preds = pd.concat([res["test_data"] for res in reg_predictions_pca], ignore_index=True)
    df_lstm_preds = pd.concat([res["test_data"] for res in reg_predictions_lstm], ignore_index=True)
    df_pca_preds[["年月日", "代號", "報酬率%", "y_pred"]].drop_duplicates(["年月日", "代號"], keep="last").to_parquet(f"{PRED_DIR}/PCA_OLS_preds.parquet")
    df_lstm_preds[["年月日", "代號", "報酬率%", "y_pred"]].drop_duplicates(["年月日", "代號"], keep="last").to_parquet(f"{PRED_DIR}/LSTM_OLS_preds.parquet")
    print("OLS 任務完成")


if __name__ == "__main__":
    main()
