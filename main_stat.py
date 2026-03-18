import pandas as pd
import numpy as np
import os
from util_stat import run_paired_wilcoxon_and_bds

if __name__ == "__main__":
    results_list = []
    
    comparisons = [
        {"algo_name": "OLS", "file_pca": "thesis_results/predictions/PCA_OLS_preds.parquet", "file_lstm": "thesis_results/predictions/LSTM_OLS_preds.parquet"},
        {"algo_name": "Ridge", "file_pca": "thesis_results/predictions/ridge_pca_preds.parquet", "file_lstm": "thesis_results/predictions/ridge_lstm_preds.parquet"},
        {"algo_name": "Lasso", "file_pca": "thesis_results/predictions/lasso_pca_preds.parquet", "file_lstm": "thesis_results/predictions/lasso_lstm_preds.parquet"},
        {"algo_name": "XGBoost", "file_pca": "thesis_results/predictions/PCA_xgb_preds.parquet", "file_lstm": "thesis_results/predictions/LSTM_xgb_preds.parquet"},
        {"algo_name": "RandomForest", "file_pca": "thesis_results/predictions/PCA_rf_preds.parquet", "file_lstm": "thesis_results/predictions/LSTM_rf_preds.parquet"}
    ]
    
    for comp in comparisons:
        res = run_paired_wilcoxon_and_bds(
            file_A=comp["file_pca"], 
            file_B=comp["file_lstm"], 
            algo_name=comp["algo_name"]
        )
        if res is not None:
            results_list.append(res)
            
    if results_list:
        df_stats = pd.DataFrame(results_list)
        save_dir = "thesis_results/statistics"
        os.makedirs(save_dir, exist_ok=True)
        excel_path = f"{save_dir}/Thesis_Final_Conclusion_Summary.xlsx"
        df_stats.to_excel(excel_path, index=False)
        
        print(f"統計檢定與結論判定全數完成！總表已匯出至: {excel_path}")
