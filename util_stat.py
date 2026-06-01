from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import wilcoxon
from statsmodels.tsa.stattools import bds

warnings.filterwarnings("ignore")


def _deduplicate_predictions(df: pd.DataFrame, name: str) -> pd.DataFrame:
    required = ["年月日", "代號", "報酬率%", "y_pred"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{name} 缺少必要欄位: {missing}")
    df = df.copy()
    df["年月日"] = pd.to_datetime(df["年月日"])
    dup = df.duplicated(["年月日", "代號"]).sum()
    if dup > 0:
        print(f"警告：{name} 有 {dup:,} 筆重複的 年月日+代號，已保留最後一筆。")
        df = df.drop_duplicates(["年月日", "代號"], keep="last")
    return df


def run_wilcoxon_test(res_model_A, res_model_B, model_a_name="PCA_Model", model_b_name="LSTM_Model"):
    df_A = _deduplicate_predictions(res_model_A["test_data"], model_a_name)
    df_B = _deduplicate_predictions(res_model_B["test_data"], model_b_name)
    merged = pd.merge(df_A, df_B, on=["年月日", "代號"], suffixes=("_A", "_B"), validate="one_to_one")
    if merged.empty:
        raise ValueError("兩個模型沒有可配對樣本。")

    pred_A = merged["y_pred_A"].values
    pred_B = merged["y_pred_B"].values
    y_true = merged["報酬率%_A"].values

    stat_pred, p_val_pred = wilcoxon(pred_A, pred_B)
    error_A = np.abs(y_true - pred_A)
    error_B = np.abs(y_true - pred_B)
    stat_err, p_val_err = wilcoxon(error_A, error_B)
    print(f"[{model_a_name} vs {model_b_name}] Wilcoxon 預測差異 p={p_val_pred:.6e}, 誤差差異 p={p_val_err:.6e}")
    return p_val_pred, p_val_err


def run_bds_test(residual_series: pd.Series) -> float:
    residual_series = pd.Series(residual_series).dropna().astype(float)
    if len(residual_series) < 20 or residual_series.nunique() <= 2:
        return np.nan
    try:
        arima_model = sm.tsa.ARIMA(residual_series, order=(1, 0, 0)).fit()
        filtered_resid = arima_model.resid.dropna()
    except Exception:
        filtered_resid = residual_series - residual_series.mean()
    try:
        _, bds_pvalue = bds(filtered_resid, max_dim=2)
        return float(np.atleast_1d(bds_pvalue)[0])
    except Exception as exc:
        print(f"BDS 檢定失敗：{exc}")
        return np.nan


def run_paired_wilcoxon_and_bds(file_A, file_B, algo_name):
    model_A_name = f"PCA_{algo_name}"
    model_B_name = f"LSTM_{algo_name}"
    print("\n" + "=" * 70)
    print(f"綜合統計與實證結論分析: {model_A_name} vs {model_B_name}")
    print("=" * 70)

    if not os.path.exists(file_A) or not os.path.exists(file_B):
        print(f"找不到 {algo_name} 的預測檔案，請確認是否已執行完該演算法。")
        return None

    df_A = _deduplicate_predictions(pd.read_parquet(file_A), model_A_name)
    df_B = _deduplicate_predictions(pd.read_parquet(file_B), model_B_name)
    merged = pd.merge(df_A, df_B, on=["年月日", "代號"], suffixes=("_A", "_B"), validate="one_to_one")
    n_samples = len(merged)
    print(f"成功配對樣本數: {n_samples:,} 筆")
    if n_samples == 0:
        return None

    pred_A = merged["y_pred_A"].values
    pred_B = merged["y_pred_B"].values
    y_true = merged["報酬率%_A"].values

    try:
        stat_pred, p_val_pred = wilcoxon(pred_A, pred_B)
    except ValueError:
        stat_pred, p_val_pred = np.nan, np.nan
    error_A = np.abs(y_true - pred_A)
    error_B = np.abs(y_true - pred_B)
    try:
        stat_err, p_val_err = wilcoxon(error_A, error_B)
    except ValueError:
        stat_err, p_val_err = np.nan, np.nan

    median_err_A = float(np.median(error_A))
    median_err_B = float(np.median(error_B))

    if pd.notna(p_val_err) and p_val_err < 0.05:
        if median_err_B < median_err_A:
            performance_status = "非線性表徵較好"
            better_model = "LSTM"
        else:
            performance_status = "線性表徵較好"
            better_model = "PCA"
    else:
        performance_status = "績效無顯著差異"
        better_model = "無顯著差異"

    # residual = actual - predicted is more standard for residual diagnostics.
    merged["residual_A"] = merged["報酬率%_A"] - merged["y_pred_A"]
    ts_resid_lin = merged.groupby("年月日")["residual_A"].mean().sort_index()
    bds_p_val = run_bds_test(ts_resid_lin)
    bds_reject_H0 = bool(pd.notna(bds_p_val) and bds_p_val < 0.05)

    print(f"[特徵表示預測差異] Wilcoxon p-value: {p_val_pred:.6e}" if pd.notna(p_val_pred) else "[特徵表示預測差異] Wilcoxon 無法計算")
    print(f"PCA 誤差中位數: {median_err_A:.6f} | LSTM 誤差中位數: {median_err_B:.6f}")
    print(f"[誤差差異] 狀態: {performance_status}, p-value: {p_val_err:.6e}" if pd.notna(p_val_err) else "[誤差差異] Wilcoxon 無法計算")
    print(f"[BDS] PCA 殘差 p-value: {bds_p_val:.6e}" if pd.notna(bds_p_val) else "[BDS] 無法計算")

    if performance_status == "非線性表徵較好" and bds_reject_H0:
        conclusion = "第一種：非線性結構存在，且非線性表示能有效利用該結構。"
    elif performance_status == "非線性表徵較好" and not bds_reject_H0:
        conclusion = "第二種：非線性表示在績效上較佳，但檢定未顯示可檢測的非線性結構。"
    elif performance_status == "線性表徵較好" and bds_reject_H0:
        conclusion = "第三種：存在非線性結構，但該結構在經濟上不可利用。"
    elif performance_status == "線性表徵較好" and not bds_reject_H0:
        conclusion = "第四種：線性假定在該市場與頻率下具備實證適用性。"
    elif performance_status == "績效無顯著差異" and bds_reject_H0:
        conclusion = "第五種：存在非線性結構，但線性表示在結構與績效上已足夠。"
    else:
        conclusion = "第六種：無非線性結構，且非線性表示表現較差或無顯著差異。"
    print(f"結論：{conclusion}")

    return {
        "演算法": algo_name,
        "配對樣本數": n_samples,
        "PCA誤差中位數": median_err_A,
        "LSTM誤差中位數": median_err_B,
        "預測差異顯著性(P-val)": p_val_pred,
        "誤差差異顯著性(P-val)": p_val_err,
        "較優特徵": better_model,
        "BDS檢定(P-val)": bds_p_val,
        "存在非線性結構": "Yes" if bds_reject_H0 else "No",
        "最終實證結論": conclusion,
    }
