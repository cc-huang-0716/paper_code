from pathlib import Path
import pandas as pd
from functools import reduce
import util_feature
from util_feature import convert_quarter
import numpy as np
from sklearn.feature_selection import VarianceThreshold

def data_processing():
# 載入數據
    root = Path(r"G:\我的雲端硬碟\碩論數據(TEJ)")
    files = list(root.rglob("*.xlsx"))
    if len(files) != 56:
        print(f"只有{len(files)}，檔案多或少了。")
    else:
        for f in files:
            print(f)


    # 合併相同特徵
    # f_list : 按年份合併的資料檔列表
    keyword = ["未調整", "Beta", "報酬", "分佈", "動向"]

    f_list = []
    for kw in keyword:
        matched_files = [f for f in files if kw in f.stem]
        f_list.append(matched_files)
        #print(matched_files)

    key_cols = ["代號", "名稱", "年月日"]
    all_dfs = []

    for file_group in f_list:
        group_dfs = []

        for f in file_group:
            df = pd.read_excel(f)
            df.columns = df.columns.str.strip()
            df["代號"] = df["代號"].astype(str).str.strip()
            df["名稱"] = df["名稱"].astype(str).str.strip()
            df["年月日"] = pd.to_datetime(df["年月日"]).dt.normalize()
            df = df.loc[:, ~df.columns.duplicated()]
            group_dfs.append(df)

        merged_group = pd.concat(group_dfs, ignore_index=True)
        merged_group = merged_group.drop_duplicates(subset=key_cols)
        all_dfs.append(merged_group)

    # 五大類特徵表再 merge
    merged_df = all_dfs[0]
    for dff in all_dfs[1:]:
        dup_cols = [c for c in dff.columns if c in merged_df.columns and c not in key_cols]
        dff = dff.drop(columns=dup_cols, errors="ignore")
        merged_df = pd.merge(merged_df, dff, on=key_cols, how="outer")

    # 總體代理變數的處理
    root_macro = Path(r"G:\我的雲端硬碟\碩論數據(TEJ)\總體代理.xlsx")
    df_macro = pd.read_excel(root_macro, header=2)
    df_macro.columns = df_macro.columns.str.strip()
    df_macro["date"] = df_macro["統計期"].apply(convert_quarter)
    df_macro["date"] = pd.to_datetime(df_macro["date"])
    df_macro = df_macro.drop(columns=["統計期"], errors="ignore")
    df_macro = df_macro.set_index("date").sort_index()

    # 用唯一日期建立 daily macro
    calendar = pd.date_range(
        merged_df["年月日"].min(),
        merged_df["年月日"].max(),
        freq="D"
    )

    macro_daily = (
        df_macro.reindex(calendar, method="ffill")
        .rename_axis("年月日")
        .reset_index()
    )

    print("merged_df rows:", len(merged_df))
    print("merged_df unique keys:", merged_df[["年月日", "代號"]].drop_duplicates().shape[0])
    print("macro_daily rows:", len(macro_daily))
    print("macro_daily unique dates:", macro_daily["年月日"].nunique())

    print("merged_df date min:", merged_df["年月日"].min())
    print("merged_df date max:", merged_df["年月日"].max())
    print("df_macro index min:", df_macro.index.min())
    print("df_macro index max:", df_macro.index.max())
    print(df_macro.head())

    # 全部合併成一個大檔案
    final_data = pd.merge(
    merged_df,
    macro_daily,
    on="年月日",
    how="left"
    )
    final_data = final_data.loc[:, ~final_data.columns.duplicated()]
    #final_data = final_data.drop(columns=['年月日'], errors = "ignore")
    print("final_data duplicated (年月日,代號):", final_data.duplicated(subset=["代號"]).sum())

    # 確認有沒有總體代理變數
    macro_cols = [
        "平均匯率(元/美元)",
        "經濟成長率(%)",
        "國內生產毛額GDP年增率(%)",
        "國民所得毛額GNI年增率(%)"
    ]

    for col in macro_cols:
        if col in final_data.columns:
            print(f"{col} 已成功合併")
        else:
            print(f"{col} 沒有合併成功")

    print(final_data.info())
    print(final_data.head())
    
    # 稀疏特徵刪除
    num_cols = final_data.select_dtypes(include=["number"]).columns
    missing_ratio = final_data[num_cols].isna().mean()
    keep_cols = missing_ratio[missing_ratio <= 0.7].index
    final_data = final_data.drop(columns=[c for c in num_cols if c not in keep_cols])

    # 高相關性特徵刪除
    corr = final_data[keep_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] >= 0.95)]
    final_data = final_data.drop(columns=to_drop)
    print("drop highly correlated:", len(to_drop))

    # 低變異欄位刪除
    num_cols = final_data.select_dtypes(include=["number"]).columns
    selector = VarianceThreshold(threshold=1e-6)
    mask = selector.fit(final_data[num_cols]).get_support()
    keep_cols = num_cols[mask]
    final_data = final_data.drop(columns=[c for c in num_cols if c not in keep_cols])

    # 補值
    num_cols = final_data.select_dtypes(include=["number"]).columns
    final_data[num_cols] = final_data[num_cols].select_dtypes(include=["number"]).ffill().fillna(final_data[num_cols].mean())
    final_data.duplicated(subset=["名稱"])

    # 輸出數據
    final_data.to_parquet(
    "final_data.parquet",
    engine="pyarrow",
    compression="snappy"
    )
    return final_data