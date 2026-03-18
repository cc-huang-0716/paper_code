# 處理總體代理變數的日期用
def convert_quarter(q):
    
    year = int(q[:3]) + 1911
    quarter = q[-2]

    if quarter == "1":
        return f"{year}-03-31"
    elif quarter == "2":
        return f"{year}-06-30"
    elif quarter == "3":
        return f"{year}-09-30"
    else:
        return f"{year}-12-31"

# Bartlett’s Test of Sphericity
from factor_analyzer import calculate_bartlett_sphericity
import pandas as pd
def bartlett(df):
    num_df = df.select_dtypes(include=["number"])
    chi_square_value, p_value = calculate_bartlett_sphericity(num_df)
    print("Chi-square:", chi_square_value)
    print("p-value:", p_value)
    return chi_square_value, p_value

# Kaiser-Meyer-Olkin
from factor_analyzer.factor_analyzer import calculate_kmo
import pandas as pd
def kaiser(df):
    num_df = df.select_dtypes(include=["float64", "int64"])
    kmo_all, kmo_model = calculate_kmo(num_df)
    kmo_table = pd.DataFrame({
                                "Variable": num_df.columns,
                                "KMO": kmo_all
                            })
    kmo_table.to_excel("kmo.xlsx")

    print("KMO overall:")
    print(kmo_model)
    print(kmo_table)
    return kmo_model, kmo_table

# Hopkins Statistic
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def hopkins(X, sample_size, random_state):
    """
    X: numpy array or pandas DataFrame
    sample_size: number of sampled points for Hopkins test
    """
    if isinstance(X, pd.DataFrame):
        X = X.values

    X = np.asarray(X, dtype=float)
    n, d = X.shape

    if sample_size >= n:
        sample_size = n - 1

    rng = np.random.default_rng(random_state)
    sample_indices = rng.choice(n, sample_size, replace=False)
    X_sample = X[sample_indices]
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    U = rng.uniform(mins, maxs, size=(sample_size, d))
    nbrs = NearestNeighbors(n_neighbors=2)
    nbrs.fit(X)
    w_dist, _ = nbrs.kneighbors(X_sample, n_neighbors=2)
    w = w_dist[:, 1]
    u_dist, _ = nbrs.kneighbors(U, n_neighbors=1)
    u = u_dist[:, 0]
    H = u.sum() / (u.sum() + w.sum())
    print("u mean:", u.mean())
    print("w mean:", w.mean())
    print("u min/max:", u.min(), u.max())
    print("w min/max:", w.min(), w.max())
    print("u sum:", u.sum())
    print("w sum:", w.sum())
    print("Hopkins:", H)
    return H

# hierarchy clustering
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster

def cluster(df):
    num_df = df.select_dtypes(include=["number"])
    Z = linkage(num_df, method='ward')
    dendrogram(Z)
    plt.show()
    labels = fcluster(Z, t=5, criterion='maxclust')
    return labels

# pca的資料前處理
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def pca_preprocessing(X):
    X = X.reset_index()
    X = X.loc[:, ~X.columns.duplicated()]
    features = X.select_dtypes(include=['float64','int64']).columns
    X[features] = (
    X.groupby("年月日")[features]
    .transform(lambda x: x.fillna(x.mean()))
    )   

    standardized_values = X.groupby("年月日")[features].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )

    X_cs = X[['代號', '年月日']].copy() 
    X_cs[features] = standardized_values
    X_cs[features] = X_cs[features].apply(pd.to_numeric, errors='coerce')
    X_cs = X_cs.sort_values(['代號', '年月日'])
    X_cs[features] = X_cs.groupby("代號")[features].fillna(method='ffill').fillna(method='bfill')
    X_cs[features] = X_cs.groupby("年月日")[features].transform(lambda x: x.fillna(x.mean()))
    X_cs[features] = X_cs[features].fillna(0)
    
    return X_cs[features]

# pca_print
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

def pca_result(pca_model, feature_names, round_name=""):

    # 解釋變異量
    exp_var = pca_model.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)
    
    # 因素負荷量表
    loadings = pd.DataFrame(
        pca_model.components_.T * np.sqrt(pca_model.explained_variance_),
        columns=[f"PC{i+1}" for i in range(pca_model.n_components_)],
        index=feature_names
    )
    
    # 存成 Excel
    loadings.to_excel(f"PCA_Factor_Loadings_{round_name}.xlsx")
    
    # 陡坡圖
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(exp_var)+1), exp_var, alpha=0.5, align='center', label='Individual')
    plt.step(range(1, len(cum_var)+1), cum_var, where='mid', label='Cumulative')
    plt.ylabel('Explained Variance Ratio')
    plt.xlabel('Principal Components')
    plt.title(f'Scree Plot - {round_name}')
    plt.legend(loc='best')
    plt.savefig(f"scree_plot_{round_name}.png", dpi=300)
    plt.close()

    return loadings



def plot_pca_loadings_heatmap(loadings, round_name="", top_n_features=15):

    plt.figure(figsize=(10, 8))
    
    # 前 5 個最重要的主成分
    top_pcs = loadings.iloc[:, :5]
    
    # 原始特徵在這 5 個 PC 中的絕對影響力總和
    importance = top_pcs.abs().sum(axis=1)
    
    # 挑出前 15 個最重要的特徵
    selected_features = importance.nlargest(top_n_features).index
    
    #  熱力圖
    sns.heatmap(top_pcs.loc[selected_features], annot=True, cmap='coolwarm', center=0, fmt=".2f")
    
    plt.title(f"PCA Factor Loadings Heatmap - {round_name}", fontsize=15)
    plt.ylabel("Original Financial Features (Top Impact)")
    plt.tight_layout()
    plt.savefig(f"PCA_Loadings_Heatmap_{round_name}.png", dpi=300)
    plt.close()


import os
import pandas as pd
import matplotlib.pyplot as plt

def save_algorithm_results(reg_predictions, algo_name="OLS"):

    # 建立輸出資料夾
    output_dir = f"thesis_results/{algo_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 數據整理
        # 績效總表
    summary_table = pd.DataFrame([res['metrics'] for res in reg_predictions])
    summary_table.index = [f"Round_{i+1}" for i in range(len(reg_predictions))]
    
        # 每日損益明細
    all_daily_df = pd.concat([res['daily_df'] for res in reg_predictions])
    
        # 因子權重 (Beta)
    all_coefs = pd.concat([res['coef'] for res in reg_predictions], axis=1).T
    all_coefs.index = summary_table.index

    # 儲存至 Excel
    excel_path = f"{output_dir}/{algo_name}_Full_Results.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        summary_table.to_excel(writer, sheet_name="Performance_Summary")
        all_daily_df.to_excel(writer, sheet_name="Daily_Returns")
        all_coefs.to_excel(writer, sheet_name="Factor_Weights")
    
    print(f"數據已存至: {excel_path}")

    # 繪製並存儲圖片
    # 這裡調用之前寫好的繪圖邏輯，但增加 savefig
    save_all_plots(all_daily_df, all_coefs, algo_name, output_dir)

def save_all_plots(daily_df, coef_df, algo_name, folder):

    # 設定字體與樣式
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
    plt.rcParams['axes.unicode_minus'] = False

    # 累積報酬曲線
    plt.figure(figsize=(12, 6))
    daily_df.index = pd.to_datetime(daily_df.index)
    oos_daily_df = daily_df[daily_df.index >= '2023-01-01'].copy()

    if not oos_daily_df.empty:
        # 計算累積淨值
        strat_cum = (1 + oos_daily_df['strategy_ret'] / 100).cumprod()
        start_date = strat_cum.index[0] - pd.Timedelta(days=1)
        strat_cum.loc[start_date] = 1.0
        strat_cum = strat_cum.sort_index()
        plt.plot(strat_cum.index, strat_cum.values, color='darkblue', label='多空對沖策略', linewidth=2)
        plt.fill_between(strat_cum.index, 1, strat_cum.values, color='skyblue', alpha=0.2)

    plt.title(f"{algo_name} - 累積淨值曲線 (2023-2025)", fontsize=16)
    plt.legend()
    plt.savefig(f"{folder}/{algo_name}_Equity_Curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    coef_df.drop('const', axis=1, errors='ignore').mean().plot(kind='bar', color='steelblue')
    plt.axhline(0, color='black', linewidth=1)
    plt.title(f"{algo_name} - 平均因子權重", fontsize=16)
    plt.savefig(f"{folder}/{algo_name}_Feature_Importance.png", dpi=300)
    plt.close()

    print(f"圖表已存入: {folder}/")

    # 權重長條圖
    plt.figure(figsize=(10, 5))
    coef_df.drop('const', axis=1, errors='ignore').mean().plot(kind='bar', color='steelblue')
    plt.axhline(0, color='black', linewidth=1)
    plt.title(f"{algo_name} - 平均因子權重 (PC1-PC10)", fontsize=16)
    plt.savefig(f"{folder}/{algo_name}_Feature_Importance.png", dpi=300)
    plt.close()

    print(f"圖表已存入: {folder}/")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_extended_analysis(daily_df, folder, algo_name="OLS"):
    # 確保索引是日期
    daily_df.index = pd.to_datetime(daily_df.index)
    daily_df['strategy_ret_decimal'] = daily_df['strategy_ret'] / 100

    # 月度報酬熱點圖
    plt.figure(figsize=(10, 4))
    monthly_ret = daily_df['strategy_ret_decimal'].resample('M').apply(lambda x: (1 + x).prod() - 1)
    pivot_table = monthly_ret.to_frame()
    pivot_table['Year'] = pivot_table.index.year
    pivot_table['Month'] = pivot_table.index.month
    heatmap_df = pivot_table.pivot(index='Year', columns='Month', values=0)
    sns.heatmap(heatmap_df, annot=True, fmt=".2%", cmap="RdYlGn", center=0)
    plt.title(f"{algo_name} - Monthly Returns Heatmap", fontsize=14)
    plt.savefig(f"{folder}/{algo_name}_Monthly_Heatmap.png", dpi=300)
    plt.close()

    # 報酬分布直方圖
    plt.figure(figsize=(10, 5))
    sns.histplot(daily_df['strategy_ret'], kde=True, color='gray', bins=50)
    plt.axvline(daily_df['strategy_ret'].mean(), color='red', linestyle='--', label=f"Mean: {daily_df['strategy_ret'].mean():.2f}%")
    plt.title(f"{algo_name} - Daily Return Distribution", fontsize=14)
    plt.xlabel("Daily Return (%)")
    plt.legend()
    plt.savefig(f"{folder}/{algo_name}_Return_Dist.png", dpi=300)
    plt.close()

    # 滾動 IC 
    plt.figure(figsize=(12, 4))
    rolling_sharpe = daily_df['strategy_ret'].rolling(window=60).mean() / daily_df['strategy_ret'].rolling(window=60).std() * np.sqrt(252)
    plt.plot(rolling_sharpe, color='orange', label='60D Rolling Sharpe')
    plt.axhline(rolling_sharpe.mean(), color='red', alpha=0.5, linestyle=':')
    plt.title(f"{algo_name} - 60-Day Rolling Sharpe Ratio", fontsize=14)
    plt.fill_between(rolling_sharpe.index, 0, rolling_sharpe, color='orange', alpha=0.1)
    plt.savefig(f"{folder}/{algo_name}_Rolling_Sharpe.png", dpi=300)
    plt.close()
    
    print(f"三張圖表已存入 {folder}")

# lstm

import pandas as pd

def lstm_preprocessing(X):

    X_copy = X.copy()
    X_copy = X_copy.loc[:, ~X_copy.columns.duplicated()]
    
    # 排除不能被標準化的標籤欄位
    features = X_copy.select_dtypes(include=['float64','int64']).columns
    features = [c for c in features if c not in ['年份', '報酬率%', '代號']]
    
    # 截面填補
    X_copy[features] = (
        X_copy.groupby("年月日")[features]
        .transform(lambda x: x.fillna(x.mean()))
    )   

    # 截面標準化
    standardized_values = X_copy.groupby("年月日")[features].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )

    # 將標準化後的值覆蓋回去
    X_cs = X_copy.copy()
    X_cs[features] = standardized_values
    
    # 時序填補
    X_cs = X_cs.sort_values(['代號', '年月日'])
    X_cs[features] = X_cs.groupby("代號")[features].fillna(method='ffill').fillna(method='bfill')
    
    # 如果還是有空值，補 0
    X_cs[features] = X_cs[features].fillna(0)
    
    return X_cs

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy


class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=10):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        
        # Encoder
        self.encoder_lstm = nn.LSTM(n_features, embedding_dim, batch_first=True)
        
        # Decoder
        self.decoder_lstm = nn.LSTM(embedding_dim, n_features, batch_first=True)
        self.decoder_output = nn.Linear(n_features, n_features)

    def forward(self, x):
        # x 形狀: (batch_size, seq_len, n_features)
        _, (hidden, _) = self.encoder_lstm(x)
        encoded = hidden.squeeze(0) # 形狀: (batch_size, embedding_dim)
        decoded_inputs = encoded.unsqueeze(1).repeat(1, self.seq_len, 1)
        decoded, _ = self.decoder_lstm(decoded_inputs)
        decoded = self.decoder_output(decoded) # 形狀: (batch_size, seq_len, n_features)
        
        return decoded, encoded

def plot_loss_curve(train_history, val_history, round_name, patience_triggered=None):

    plt.figure(figsize=(10, 6))
    plt.plot(train_history, label='Train Loss', color='blue', linewidth=2)
    plt.plot(val_history, label='Validation Loss', color='orange', linewidth=2)
    
    if patience_triggered is not None:
        best_epoch = len(train_history) - patience_triggered - 1
        plt.axvline(x=best_epoch, color='red', linestyle='--', label=f'Best Model (Epoch {best_epoch+1})')
        plt.scatter(best_epoch, val_history[best_epoch], color='red', zorder=5)

    plt.title(f"LSTM Autoencoder Training Loss - {round_name}", fontsize=15)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs("loss_curves", exist_ok=True)
    plt.savefig(f"loss_curves/Loss_Curve_{round_name}.png", dpi=300)
    plt.close()

def create_lstm_sequences(df, features_cols, time_steps=5, predict_horizon=1):

    X_list = []
    feature_index_list = []
    target_index_list = []
    
    df = df.sort_values(['代號', '年月日']).copy()
    df[features_cols] = df[features_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    for ticker, group in df.groupby('代號'):
        group_values = group[features_cols].values.astype(np.float32)
        group_indices = group.index.values

        for i in range(len(group) - time_steps - predict_horizon + 1):
            X_list.append(group_values[i : i + time_steps])
            feature_index_list.append(group_indices[i + time_steps - 1])      
            target_index_list.append(group_indices[i + time_steps - 1 + predict_horizon])  
            
    return np.array(X_list, dtype=np.float32), feature_index_list, target_index_list

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy
import matplotlib.pyplot as plt
import os
from torch.utils.tensorboard import SummaryWriter 

def plot_loss_curve(train_history, val_history, round_name, patience_triggered=None):

    plt.figure(figsize=(10, 6))
    plt.plot(train_history, label='Train Loss', color='blue', linewidth=2)
    plt.plot(val_history, label='Validation Loss', color='orange', linewidth=2)
    
    # 標示出 Early Stopping 觸發的最佳權重點
    if patience_triggered is not None:
        best_epoch = len(train_history) - patience_triggered - 1
        plt.axvline(x=best_epoch, color='red', linestyle='--', label=f'Best Model (Epoch {best_epoch+1})')
        plt.scatter(best_epoch, val_history[best_epoch], color='red', zorder=5)

    plt.title(f"LSTM Autoencoder Training Loss - {round_name}", fontsize=15)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs("loss_curves", exist_ok=True)
    plt.savefig(f"loss_curves/Loss_Curve_{round_name}.png", dpi=300)
    plt.close()


def train_and_extract_pytorch(X_train, X_val, X_test=None, round_name="Round_X", encoding_dim=10, epochs=50, batch_size=128, patience=5, return_model_only=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(train_tensor, train_tensor) 
    val_dataset = TensorDataset(val_tensor, val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    seq_len = X_train.shape[1]
    n_features = X_train.shape[2]
    
    model = LSTMAutoencoder(seq_len, n_features, encoding_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 記錄器初始化 
    train_history = []
    val_history = []
    writer = SummaryWriter(log_dir=f"runs/LSTM_{round_name}") 
    
    best_val_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0

    print(f"啟動訓練 ({round_name}),在終端機輸入 'tensorboard --logdir=runs' 監控")
    
    for epoch in range(epochs):

        # 訓練階段
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            decoded, _ = model(batch_x)
            loss = criterion(decoded, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= len(train_loader.dataset)

        # 驗證階段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                decoded, _ = model(batch_x)
                loss = criterion(decoded, batch_y)
                val_loss += loss.item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)

        # 紀錄歷史軌跡
        train_history.append(train_loss)
        val_history.append(val_loss)
        
        # TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)

        print(f"Epoch {epoch+1:02d}/{epochs} - Train Loss: {train_loss:.5f} - Val Loss: {val_loss:.5f}")

        # Early Stopping 檢查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0 
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f" 連續 {patience} 次 Val Loss 未下降")
                break
                
    writer.close() 
    
    # 畫出並儲存學術 Loss 曲線圖
    plot_loss_curve(train_history, val_history, round_name, patience_triggered=patience_counter)

    # 載入最佳權重
    model.load_state_dict(best_model_weights)
    
    # 回傳模型即可
    if return_model_only:
        return model


import torch

# 使用 lstm 參數用
def load_and_extract_features(X_3d, seq_len, n_features, model_path, encoding_dim=10):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 建立空的模型架構
    model = LSTMAutoencoder(seq_len, n_features, encoding_dim).to(device)
    
    # 載入訓練好的權重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() 
    
    # 轉換資料並預測
    X_tensor = torch.tensor(X_3d, dtype=torch.float32).to(device)
    with torch.no_grad():
        _, encoded_features = model(X_tensor)
        
    return encoded_features.cpu().numpy()
