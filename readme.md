# Financial Time Series Prediction: PCA vs. LSTM Feature Representation

本專案整理自金融時間序列預測與資產定價實證研究程式碼，主要目標是比較線性特徵表示與非線性特徵表示在台股橫斷面報酬預測上的差異。

研究流程以台灣上市櫃公司日資料為基礎，使用財務與市場特徵預測下一期個股報酬率，並進一步建構 long-short 投資組合進行樣本外績效評估。
本專案的核心比較為：

* **PCA**：作為線性降維與特徵表示方法
* **LSTM Autoencoder**：作為非線性序列表徵方法

接著將兩種特徵表示分別輸入多種監督式模型，包括 OLS、Ridge、Lasso、Random Forest 與 XGBoost，並比較其預測能力、投資績效與統計檢定結果。

> 本專案僅供學術研究、論文實證與作品集展示使用，不構成任何投資建議。

---

## Project Overview

本專案主要包含以下功能：

* 金融時間序列資料前處理
* PCA 特徵萃取
* LSTM Autoencoder 非線性表徵學習
* 多種預測模型訓練與比較
* 橫斷面 long-short 投資組合回測
* Rank IC、Sharpe Ratio、Max Drawdown 等績效指標計算
* Wilcoxon paired test 比較 PCA 與 LSTM 預測差異
* BDS test 檢定殘差是否存在非線性結構
* 匯出 Excel 結果、預測檔案與圖表

---

## Repository Structure

```text
.
├── main_pretest.py
├── train_lstm.py
├── main_ols.py
├── main_ridge.py
├── main_rf.py
├── main_xgboost.py
├── main_stat.py
├── ttest.py
├── util_algorithms.py
├── util_feature.py
├── util_stat.py
└── README.md
```

---

## File Description

### `main_pretest.py`

執行 PCA 前置檢定，用於確認資料是否適合進行降維分析。

主要功能包含：

* 讀取 `final_data.parquet`
* 篩選數值型特徵
* 移除全空值與低變異特徵
* 執行 Bartlett's Test
* 執行 KMO 檢定
* 執行 Hopkins Statistic
* 匯出 PCA 前置檢定結果

輸出路徑：

```text
thesis_results/pretest/
```

---

### `train_lstm.py`

訓練 LSTM Autoencoder，用於將多期金融特徵轉換為非線性序列表徵。

主要功能包含：

* 讀取 `final_data.parquet`
* 依年份切分訓練集與驗證集
* 建立 5 期時間序列資料
* 訓練 LSTM Autoencoder
* 使用 early stopping 控制訓練
* 儲存各輪次的 LSTM encoder 權重

輸出路徑：

```text
saved_models/
```

模型命名格式：

```text
lstm_encoder_Val_1.pth
lstm_encoder_Val_2.pth
lstm_encoder_Val_3.pth
lstm_encoder_Final_Test.pth
```

---

### `main_ols.py`

執行 OLS regression 實驗，比較 PCA 特徵與 LSTM 特徵在傳統線性迴歸模型下的預測與投資績效。

主要功能包含：

* PCA 特徵建立
* LSTM encoder 特徵載入
* OLS 模型訓練
* 樣本外預測
* long-short 投資組合回測
* 匯出績效指標與預測結果

輸出包含：

```text
thesis_results/OLS_Baseline_PCA/
thesis_results/OLS_Experimental_LSTM/
thesis_results/predictions/PCA_OLS_preds.parquet
thesis_results/predictions/LSTM_OLS_preds.parquet
```

---

### `main_ridge.py`

執行 Ridge 與 Lasso regression 實驗，並同時比較 PCA 與 LSTM 兩種特徵表示。

主要功能包含：

* PCA + Ridge
* LSTM + Ridge
* PCA + Lasso
* LSTM + Lasso
* long-short 策略績效計算
* 匯出模型結果與預測檔案

輸出包含：

```text
thesis_results/Ridge_Baseline_PCA/
thesis_results/Ridge_Experimental_LSTM/
thesis_results/Lasso_Baseline_PCA/
thesis_results/Lasso_Experimental_LSTM/
thesis_results/predictions/ridge_pca_preds.parquet
thesis_results/predictions/ridge_lstm_preds.parquet
thesis_results/predictions/lasso_pca_preds.parquet
thesis_results/predictions/lasso_lstm_preds.parquet
```

---

### `main_rf.py`

執行 Random Forest 實驗，用於比較非線性機器學習模型在 PCA 與 LSTM 特徵上的表現。

主要功能包含：

* PCA 特徵建立
* LSTM encoder 特徵載入
* Random Forest Regressor 訓練
* 樣本外預測
* long-short 回測
* 匯出結果與預測檔案

輸出包含：

```text
thesis_results/RandomForest_Baseline_PCA/
thesis_results/RandomForest_Experimental_LSTM/
thesis_results/predictions/PCA_rf_preds.parquet
thesis_results/predictions/LSTM_rf_preds.parquet
```

---

### `main_xgboost.py`

執行 XGBoost 實驗，用於比較梯度提升樹模型在 PCA 與 LSTM 特徵上的預測能力。

主要功能包含：

* PCA 特徵建立
* LSTM encoder 特徵載入
* XGBoost Regressor 訓練
* 樣本外預測
* long-short 策略回測
* 匯出結果與預測檔案

輸出包含：

```text
thesis_results/XGBoost_Baseline_PCA/
thesis_results/XGBoost_Experimental_LSTM/
thesis_results/predictions/PCA_xgb_preds.parquet
thesis_results/predictions/LSTM_xgb_preds.parquet
```

---

### `main_stat.py`

執行最終統計檢定，用於比較 PCA 與 LSTM 特徵表示的預測差異與非線性結構。

主要功能包含：

* 讀取各模型的 PCA 與 LSTM 預測結果
* 進行 paired Wilcoxon signed-rank test
* 比較 PCA 與 LSTM 的預測差異與誤差差異
* 執行 BDS test 檢定 PCA 殘差是否存在非線性結構
* 依照統計結果產出實證結論

輸出路徑：

```text
thesis_results/statistics/Thesis_Final_Conclusion_Summary.xlsx
```

---

### `ttest.py`

輔助檢定腳本，用於 KMO 與低變異欄位篩選測試。

主要功能包含：

* 讀取 `final_data.parquet`
* 篩選數值型資料
* 移除低變異欄位
* 執行 KMO 檢定
* 匯出檢定結果

---

### `util_algorithms.py`

模型訓練、預測與回測工具函式。

主要功能包含：

* OLS regression
* Ridge regression
* Lasso regression
* Random Forest regression
* XGBoost regression
* long-short 投資組合建構
* Daily Rank IC 計算
* 累積報酬率計算
* 年化報酬率、年化波動度與 Sharpe Ratio 計算
* 最大回撤與勝率計算
* 模型係數與特徵重要度整理

---

### `util_feature.py`

特徵工程、PCA、LSTM 與圖表輸出工具函式。

主要功能包含：

* PCA 前處理
* LSTM 前處理
* 建立時間序列樣本
* LSTM Autoencoder 定義與訓練
* 載入 LSTM encoder 並萃取特徵
* PCA factor loading 輸出
* Scree plot 繪製
* PCA loading heatmap 繪製
* 投資組合績效圖表輸出
* Equity curve
* Drawdown chart
* Monthly return heatmap
* Return distribution
* Rolling Sharpe
* Rolling Rank IC
* Feature importance

---

### `util_stat.py`

統計檢定工具函式。

主要功能包含：

* 預測結果去重
* PCA 與 LSTM 預測結果配對
* Wilcoxon signed-rank test
* 殘差序列處理
* BDS test
* 自動產出線性與非線性結構比較結論

---

## Methodology

### 1. Data Splitting

本專案使用 rolling / expanding style 的年份切分方式進行樣本外測試：

```text
Val_1      Train: 2015–2019    Test: 2020
Val_2      Train: 2016–2020    Test: 2021
Val_3      Train: 2017–2021    Test: 2022
Final_Test Train: 2015–2022    Test: 2023–2025
```

---

### 2. Feature Representation

本研究比較兩種特徵表示方式：

#### PCA Features

PCA 作為線性降維方法，將原始金融特徵轉換為主要成分。

```text
Original Features → Cross-sectional Standardization → PCA → PC1, PC2, ..., PC10
```

#### LSTM Autoencoder Features

LSTM Autoencoder 作為非線性序列表徵方法，將多期特徵序列壓縮為低維 embedding。

```text
Original Features → 5-period Sequence → LSTM Autoencoder → LSTM_F1, ..., LSTM_F10
```

---

### 3. Prediction Models

兩種特徵表示皆會輸入相同的監督式模型進行比較：

* OLS
* Ridge
* Lasso
* Random Forest
* XGBoost

預測目標為下一期個股報酬率：

```text
target_ret = next-period stock return
```

---

### 4. Portfolio Backtesting

模型預測結果會進一步轉換成橫斷面 long-short 投資組合：

```text
Long Portfolio  = Top 10% predicted return stocks
Short Portfolio = Bottom 10% predicted return stocks
Strategy Return = Long Return - Short Return
```

主要績效指標包含：

* Cumulative Return
* Annual Return
* Annual Volatility
* Sharpe Ratio
* Maximum Drawdown
* Win Rate
* Pearson IC
* Spearman Rank IC
* Daily Rank IC

---

### 5. Statistical Testing

本專案使用兩種統計檢定輔助判斷線性與非線性特徵表示的差異。

#### Wilcoxon Signed-Rank Test

用於比較 PCA 與 LSTM 的預測值差異與絕對誤差差異。

#### BDS Test

用於檢查 PCA 模型殘差是否仍存在可檢測的非線性結構。

根據 Wilcoxon 與 BDS 結果，程式會自動整理出不同類型的實證結論。

---

## Requirements

建議使用 Python 3.9 以上版本。

主要套件包含：

```text
pandas
numpy
scikit-learn
statsmodels
scipy
matplotlib
seaborn
torch
xgboost
factor_analyzer
openpyxl
pyarrow
tensorboard
```

可使用以下指令安裝：

```bash
pip install pandas numpy scikit-learn statsmodels scipy matplotlib seaborn torch xgboost factor_analyzer openpyxl pyarrow tensorboard
```

---

## Data Requirement

本專案預設資料檔案名稱為：

```text
final_data.parquet
```

資料中至少需包含以下欄位：

```text
年月日
年份
代號
名稱
報酬率%
```

其餘數值型欄位會被視為模型特徵。

由於原始資料可能涉及資料授權或研究限制，完整資料檔案不一定包含於此 repo 中。
若要執行程式，請將 `final_data.parquet` 放置於專案根目錄。

---

## Suggested Running Order

建議依照以下順序執行：

### 1. PCA Pretest

```bash
python main_pretest.py
```

### 2. Train LSTM Autoencoder

```bash
python train_lstm.py
```

### 3. Run Prediction Models

```bash
python main_ols.py
python main_ridge.py
python main_rf.py
python main_xgboost.py
```

### 4. Run Statistical Tests

```bash
python main_stat.py
```

---

## Output

主要輸出資料夾如下：

```text
saved_models/
thesis_results/
├── pretest/
├── pca/
├── predictions/
├── statistics/
├── OLS_Baseline_PCA/
├── OLS_Experimental_LSTM/
├── Ridge_Baseline_PCA/
├── Ridge_Experimental_LSTM/
├── Lasso_Baseline_PCA/
├── Lasso_Experimental_LSTM/
├── RandomForest_Baseline_PCA/
├── RandomForest_Experimental_LSTM/
├── XGBoost_Baseline_PCA/
└── XGBoost_Experimental_LSTM/
```

常見輸出包含：

* 模型預測結果 `.parquet`
* 模型績效總表 `.xlsx`
* 每日策略報酬
* 因子權重與特徵重要度
* PCA factor loading
* Scree plot
* Equity curve
* Drawdown chart
* Monthly return heatmap
* Rolling Sharpe
* Rolling Rank IC
* Wilcoxon / BDS 統計檢定結果

---

## Notes

本專案為研究與論文實證用途，因此部分設定仍保留於程式碼中，例如：

* 訓練與測試年份
* PCA components 數量
* LSTM time steps
* LSTM encoding dimension
* Random Forest 與 XGBoost 參數
* long-short 投資組合比例

若要進一步提升可維護性，建議將上述設定整理為 config file，例如：

```text
config.yaml
```

---

## Limitations

本專案仍有以下限制：

* 預測目標為短期個股報酬，資料雜訊高
* 模型績效容易受到樣本期間與市場狀態影響
* LSTM Autoencoder 產生的非線性表徵不保證能轉化為經濟上可利用的超額報酬
* 回測流程為研究用途，未完整納入交易成本、滑價、融券限制與實際可交易性
* 原始資料與部分結果檔案未必包含於 repo 中
* 程式目前以研究腳本為主，尚未整理為完整 Python package

---

## Future Improvements

未來可進一步改善方向包括：

* 將資料路徑、模型參數與年份切分集中至 config file
* 將 PCA / LSTM 特徵建立流程模組化
* 加入交易成本與滑價假設
* 加入不同投組分組方式，例如 top 5%、top 20%、long-only
* 加入更多模型，例如 Elastic Net、LightGBM、CatBoost
* 加入 feature selection 與穩健性檢定
* 補充完整實驗結果表格與圖表
* 建立一鍵執行 pipeline
* 加入單元測試與環境設定檔

---

## Disclaimer

This repository is for academic research and portfolio demonstration purposes only.
The results should not be interpreted as financial advice, investment recommendations, or trading signals.
Any investment decision should be made based on independent research and proper risk assessment.
