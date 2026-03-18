import pandas as pd
import numpy as np
from util_feature import kaiser
import time
from util_feature import pca_processing
from dataset import data_processing

#df = data_processing()
df = pd.read_parquet("final_data.parquet")
print(df.columns)

num_df = df.select_dtypes(include=["float64", "int64"])
print("原始 numeric shape:", num_df.shape)

from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=1e-6)
num_df = num_df.loc[:, selector.fit(num_df).get_support()]
num_df.shape

start = time.time()
kaiser(df=df)
print("檢定完成", df.shape, "耗時", time.time() - start,"秒")



