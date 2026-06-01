import os
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
from util_feature import bartlett, kaiser, hopkins

DATA_PATH = "final_data.parquet"
OUT_DIR = "thesis_results/pretest"
EXCLUDE_LIST = ["報酬率%", "報酬率％", "年月日", "年份", "代號", "名稱", "index", "target_ret"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    start = time.time()
    parq = pd.read_parquet(DATA_PATH)
    parq.columns = parq.columns.str.strip().str.replace("％", "%")
    numeric_cols = parq.select_dtypes(include=["number"]).columns
    feat_cols = [c for c in numeric_cols if c not in EXCLUDE_LIST]
    num_df = parq[feat_cols].replace([float("inf"), float("-inf")], pd.NA).dropna(axis=1, how="all")
    num_df = num_df.fillna(num_df.mean()).fillna(0)
    # Remove constant columns before Bartlett/KMO/Hopkins.
    num_df = num_df.loc[:, num_df.nunique(dropna=True) > 1]
    print(f"讀檔完成 {parq.shape}，pretest 特徵數={num_df.shape[1]}，耗時 {time.time() - start:.2f} 秒")

    rows = []
    start = time.time()
    chi_square, bartlett_p = bartlett(num_df)
    rows.append({"檢定": "Bartlett's Test", "統計量": chi_square, "p_value或指標值": bartlett_p})
    print("Bartlett 完成，耗時", time.time() - start, "秒")

    start = time.time()
    kmo_model, kmo_table = kaiser(num_df, output_path=f"{OUT_DIR}/kmo_detail.xlsx")
    rows.append({"檢定": "KMO", "統計量": None, "p_value或指標值": kmo_model})
    print("KMO 完成，耗時", time.time() - start, "秒")

    start = time.time()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(num_df.values)
    # Hopkins is O(n^2-ish); cap sample size to avoid overlong runs.
    sample_size = min(5000, max(100, len(num_df) // 10))
    H = hopkins(X_scaled, sample_size=sample_size, random_state=42)
    rows.append({"檢定": "Hopkins Statistic", "統計量": None, "p_value或指標值": H})
    print("Hopkins 完成，耗時", time.time() - start, "秒")

    summary = pd.DataFrame(rows)
    summary.to_excel(f"{OUT_DIR}/PCA_Pretest_Summary.xlsx", index=False)
    print(f"預檢結果已輸出至 {OUT_DIR}/PCA_Pretest_Summary.xlsx")


if __name__ == "__main__":
    main()
