import pandas as pd
import time
from dataset import data_processing
from util_feature import bartlett
from util_feature import kaiser
from util_feature import hopkins
from sklearn.preprocessing import StandardScaler

# 載入數據 
start = time.time()
parq = pd.read_parquet("final_data.parquet")
#parq = data_processing()
num_df = parq.select_dtypes(include=["number"]).dropna()
parq.columns = parq.columns.str.strip().str.replace('％', '%')
print("讀檔完成", parq.shape, "耗時", time.time() - start,"秒")

# Bartlett’s Test of Sphericity
start = time.time()
bartlett(df=num_df)
print("檢定完成", num_df.shape, "耗時", time.time() - start,"秒")

# Kaiser–Meyer–Olkin  test
start = time.time()
kaiser(df=num_df)
print("檢定完成", num_df.shape, "耗時", time.time() - start,"秒")

# Hopkins Statistic
start = time.time()
X = num_df.values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
H = hopkins(X=X_scaled, sample_size=87713, random_state=42)
print("Hopkins statistic:", H)
print("檢定完成", num_df.shape, "耗時", time.time() - start,"秒")

# hierarchy clustering
#start = time.time()
#cluster(df=num_df)
#print("檢定完成", num_df.shape, "耗時", time.time() - start)