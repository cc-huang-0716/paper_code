"""Small helper script for KMO pre-check.

The original file imported pca_processing, but util_feature.py does not define that function.
This fixed version keeps only the KMO/low-variance check logic.
"""

import time
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from util_feature import kaiser

DATA_PATH = "final_data.parquet"


def main():
    df = pd.read_parquet(DATA_PATH)
    df.columns = df.columns.str.strip().str.replace("％", "%")
    num_df = df.select_dtypes(include=["number"]).replace([float("inf"), float("-inf")], pd.NA)
    num_df = num_df.fillna(num_df.mean()).fillna(0)
    print("原始 numeric shape:", num_df.shape)

    selector = VarianceThreshold(threshold=1e-6)
    selected = selector.fit(num_df).get_support()
    num_df = num_df.loc[:, selected]
    print("移除低變異欄位後 shape:", num_df.shape)

    start = time.time()
    kaiser(df=num_df, output_path="kmo_ttest.xlsx")
    print("檢定完成", num_df.shape, "耗時", time.time() - start, "秒")


if __name__ == "__main__":
    main()
