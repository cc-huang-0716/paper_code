from scipy.stats import wilcoxon
import numpy as np

def run_wilcoxon_test(res_model_A, res_model_B, model_a_name="PCA_Model", model_b_name="LSTM_Model"):
    
    pred_A = res_model_A['test_data']['y_pred'].values
    pred_B = res_model_B['test_data']['y_pred'].values
    y_true = res_model_A['test_data']['報酬率%'].values

    # 預測報酬差異
    stat_pred, p_val_pred = wilcoxon(pred_A, pred_B)
    
    # 絕對誤差差異
    error_A = np.abs(y_true - pred_A)
    error_B = np.abs(y_true - pred_B)
    stat_err, p_val_err = wilcoxon(error_A, error_B)

    # 輸出結果
    print(f"\n[{model_a_name} vs {model_b_name}] Wilcoxon 檢定結果：")
    print("-" * 50)
    
    print(f"預測報酬差異檢定 | Statistic: {stat_pred:.2f}, P-value: {p_val_pred:.6e}")
    if p_val_pred < 0.05:
        print("結論：兩種特徵表示法產生的預測報酬具有顯著系統性差異。")

    print(f"絕對誤差差異檢定 | Statistic: {stat_err:.2f}, P-value: {p_val_err:.6e}")
    if p_val_err < 0.05:
        if np.median(error_A) > np.median(error_B):
            print(f"結論：誤差有顯著差異，且 {model_b_name} 預測更精準。")
        else:
            print(f"結論：誤差有顯著差異，且 {model_a_name} 預測更精準。")

    return p_val_pred, p_val_err

import pandas as pd
import numpy as np
import os
from scipy.stats import wilcoxon
import statsmodels.api as sm
from statsmodels.tsa.stattools import bds
import warnings
warnings.filterwarnings('ignore')

def run_bds_test(residual_series):

    # 套用 ARIMA 模型過濾線性結構
    try:
        arima_model = sm.tsa.ARIMA(residual_series, order=(1, 0, 0)).fit()
        filtered_resid = arima_model.resid
    except:
        # 若 ARIMA 無法收斂，退回單純去均值
        filtered_resid = residual_series - residual_series.mean()

    # 對過濾後的殘差執行 BDS 檢定
    bds_stat, bds_pvalue = bds(filtered_resid, max_dim=2)
    p_val = np.atleast_1d(bds_pvalue)[0]
    
    # 回傳維度 2 的 p-value
    return float(p_val)

def run_paired_wilcoxon_and_bds(file_A, file_B, algo_name):

    model_A_name = f"PCA_{algo_name}"
    model_B_name = f"LSTM_{algo_name}"
    
    print(f"\n" + "="*70)
    print(f"綜合統計與實證結論分析: {model_A_name} vs {model_B_name}")
    print("="*70)
    
    # 檢查檔案是否存在
    if not os.path.exists(file_A) or not os.path.exists(file_B):
        print(f"找不到 {algo_name} 的預測檔案，請確認是否已執行完該演算法。")
        return None
        
    df_A = pd.read_parquet(file_A)
    df_B = pd.read_parquet(file_B)
    
    # 嚴格對齊
    merged = pd.merge(
        df_A, df_B, 
        on=['年月日', '代號'], 
        suffixes=('_A', '_B')
    )
    n_samples = len(merged)
    print(f"成功配對樣本數: {n_samples:,} 筆")
    if n_samples == 0:
        return None

    # 提取對齊後的數值
    pred_A = merged['y_pred_A'].values
    pred_B = merged['y_pred_B'].values
    y_true = merged['報酬率%_A'].values 
    
    # Wilcoxon 檢定
    stat_pred, p_val_pred = wilcoxon(pred_A, pred_B)
    
    error_A = np.abs(y_true - pred_A)
    error_B = np.abs(y_true - pred_B)
    stat_err, p_val_err = wilcoxon(error_A, error_B)
    
    median_err_A = np.median(error_A)
    median_err_B = np.median(error_B)
    
    # 判定哪種表徵績效較好
    if p_val_err < 0.05:
        if median_err_B < median_err_A:
            performance_status = "非線性表徵較好"
            better_model = "LSTM"
        else:
            performance_status = "線性表徵較好"
            better_model = "PCA"
    else:
        performance_status = "績效無顯著差異"
        better_model = "無顯著差異"

    # BDS 檢定
    # 計算線性模型的殘差
    merged['residual_A'] = merged['y_pred_A'] - merged['報酬率%_A']
    
    # 每日橫截面的平均殘差
    ts_resid_lin = merged.groupby('年月日')['residual_A'].mean().sort_index()
    
    # 執行 BDS 檢定
    bds_p_val = run_bds_test(ts_resid_lin)
    bds_reject_H0 = bds_p_val < 0.05

    # 綜合判定
    print(f"\n[假說一：特徵表示法是否造成預測結果的系統性差異？]")
    print(f"檢定統計量: {stat_pred:.2f}, Wilcoxon P-value: {p_val_pred:.6e}")
    
    print(f"\n[假說二：何種特徵表示法之預測精準度較高？]")
    print(f"PCA 誤差中位數: {median_err_A:.6f} | LSTM 誤差中位數: {median_err_B:.6f}")
    print(f"狀態：【{performance_status}】 (Wilcoxon P-value: {p_val_err:.6e})")

    print(f"\n[假說三：市場是否存在線性模型無法捕捉之非線性結構？]")
    print(f"線性模型殘差 BDS P-value: {bds_p_val:.6e}")
    if bds_reject_H0:
        print("狀態：【拒絕 H0】,存在非線性結構。")
    else:
        print("狀態：【不拒絕 H0】,無顯著非線性結構。")

    print(f"[最終實證結論探討]")
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
        conclusion = "第六種：無非線性結構，且非線性表示反而表現較差或無顯著差異。"
        
    print(f"結論：{conclusion}")

    # 整合進 Excel 大表
    return {
        "演算法": algo_name,
        "配對樣本數": n_samples,
        "PCA誤差中位數": median_err_A,
        "LSTM誤差中位數": median_err_B,
        "誤差差異顯著性(P-val)": p_val_err,
        "較優特徵": better_model,
        "BDS檢定(P-val)": bds_p_val,
        "存在非線性結構": "Yes" if bds_reject_H0 else "No",
        "最終實證結論": conclusion.split(":")[0]
    }