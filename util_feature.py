"""Feature preprocessing, PCA/LSTM utilities, and thesis figure helpers.

Important return convention:
- create_lstm_sequences returns three objects:
  (X_3d, feature_index_list, target_index_list)
"""

from __future__ import annotations

import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    torch = None
    nn = None
    optim = None
    DataLoader = None
    TensorDataset = None
    SummaryWriter = None


def convert_quarter(q: str) -> str:
    year = int(q[:3]) + 1911
    quarter = q[-1]
    return {
        "1": f"{year}-03-31",
        "2": f"{year}-06-30",
        "3": f"{year}-09-30",
        "4": f"{year}-12-31",
    }.get(quarter, f"{year}-12-31")


def _numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    num_df = df.select_dtypes(include=["number"]).copy()
    num_df = num_df.replace([np.inf, -np.inf], np.nan)
    num_df = num_df.dropna(axis=1, how="all")
    # Constant columns can break KMO/Bartlett; remove them.
    nunique = num_df.nunique(dropna=True)
    num_df = num_df.loc[:, nunique > 1]
    num_df = num_df.fillna(num_df.mean()).fillna(0)
    return num_df


def bartlett(df: pd.DataFrame):
    from factor_analyzer import calculate_bartlett_sphericity
    num_df = _numeric_df(df)
    chi_square_value, p_value = calculate_bartlett_sphericity(num_df)
    print("Chi-square:", chi_square_value)
    print("p-value:", p_value)
    return chi_square_value, p_value


def kaiser(df: pd.DataFrame, output_path: str = "kmo.xlsx"):
    from factor_analyzer.factor_analyzer import calculate_kmo
    num_df = _numeric_df(df)
    kmo_all, kmo_model = calculate_kmo(num_df)
    kmo_table = pd.DataFrame({"Variable": num_df.columns, "KMO": kmo_all})
    kmo_table.to_excel(output_path, index=False)
    print("KMO overall:", kmo_model)
    return kmo_model, kmo_table


def hopkins(X, sample_size: int = 1000, random_state: int = 42) -> float:
    from sklearn.neighbors import NearestNeighbors
    if isinstance(X, pd.DataFrame):
        X = X.values
    X = np.asarray(X, dtype=float)
    X = X[~np.isnan(X).any(axis=1)]
    n, d = X.shape
    if n < 3:
        raise ValueError("Hopkins statistic requires at least 3 observations.")
    sample_size = min(sample_size, n - 1)

    rng = np.random.default_rng(random_state)
    sample_indices = rng.choice(n, sample_size, replace=False)
    X_sample = X[sample_indices]
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    U = rng.uniform(mins, maxs, size=(sample_size, d))

    nbrs = NearestNeighbors(n_neighbors=2).fit(X)
    w_dist, _ = nbrs.kneighbors(X_sample, n_neighbors=2)
    w = w_dist[:, 1]
    u_dist, _ = nbrs.kneighbors(U, n_neighbors=1)
    u = u_dist[:, 0]
    H = u.sum() / (u.sum() + w.sum())
    print("Hopkins:", H)
    return float(H)


def cluster(df: pd.DataFrame, n_clusters: int = 5):
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    num_df = _numeric_df(df)
    Z = linkage(num_df, method="ward")
    dendrogram(Z)
    plt.tight_layout()
    plt.show()
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    return labels


def _feature_columns(df: pd.DataFrame, exclude_cols: list[str] | None = None) -> list[str]:
    exclude_cols = set(exclude_cols or [])
    numeric_cols = df.select_dtypes(include=["number"]).columns
    return [c for c in numeric_cols if c not in exclude_cols]


def pca_preprocessing(X: pd.DataFrame, exclude_cols: list[str] | None = None) -> pd.DataFrame:
    """Cross-sectionally standardize numeric features by date and fill missing values.

    Returns only numeric feature columns, aligned to the input row order after reset_index.
    """
    X = X.reset_index(drop=True).copy()
    X = X.loc[:, ~X.columns.duplicated()]
    default_exclude = ["報酬率%", "報酬率％", "年月日", "年份", "代號", "名稱", "index", "target_ret"]
    features = _feature_columns(X, exclude_cols=(exclude_cols or default_exclude))
    if "年月日" not in X.columns or "代號" not in X.columns:
        raise KeyError("pca_preprocessing requires columns '年月日' and '代號'.")

    X[features] = X[features].apply(pd.to_numeric, errors="coerce")
    X[features] = X.groupby("年月日")[features].transform(lambda s: s.fillna(s.mean()))
    standardized = X.groupby("年月日")[features].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-8))

    X_cs = X[["代號", "年月日"]].copy()
    X_cs[features] = standardized
    X_cs = X_cs.sort_values(["代號", "年月日"])
    X_cs[features] = X_cs.groupby("代號")[features].ffill().bfill()
    X_cs[features] = X_cs.groupby("年月日")[features].transform(lambda s: s.fillna(s.mean()))
    X_cs[features] = X_cs[features].fillna(0)
    # Restore original index order after sort/fill
    X_cs = X_cs.sort_index()
    return X_cs[features]


def pca_result(pca_model, feature_names, round_name: str = "", output_dir: str = "thesis_results/pca") -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    exp_var = pca_model.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)
    loadings = pd.DataFrame(
        pca_model.components_.T * np.sqrt(pca_model.explained_variance_),
        columns=[f"PC{i + 1}" for i in range(pca_model.n_components_)],
        index=feature_names,
    )
    loadings.to_excel(os.path.join(output_dir, f"PCA_Factor_Loadings_{round_name}.xlsx"))

    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(exp_var) + 1), exp_var, alpha=0.65, align="center", label="Individual")
    plt.step(range(1, len(cum_var) + 1), cum_var, where="mid", label="Cumulative")
    plt.ylabel("Explained Variance Ratio")
    plt.xlabel("Principal Components")
    plt.title(f"Scree Plot - {round_name}")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"scree_plot_{round_name}.png"), dpi=300)
    plt.close()
    return loadings


def plot_pca_loadings_heatmap(loadings: pd.DataFrame, round_name: str = "", top_n_features: int = 15, output_dir: str = "thesis_results/pca") -> str:
    if sns is None:
        raise ImportError("seaborn is required for heatmap plotting.")
    os.makedirs(output_dir, exist_ok=True)
    top_pcs = loadings.iloc[:, : min(5, loadings.shape[1])]
    importance = top_pcs.abs().sum(axis=1)
    selected = importance.nlargest(min(top_n_features, len(importance))).index
    plt.figure(figsize=(10, max(6, 0.35 * len(selected))))
    sns.heatmap(top_pcs.loc[selected], annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title(f"PCA Factor Loadings Heatmap - {round_name}")
    plt.ylabel("Original Features")
    plt.tight_layout()
    path = os.path.join(output_dir, f"PCA_Loadings_Heatmap_{round_name}.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def save_algorithm_results(reg_predictions: list[dict], algo_name: str = "OLS", output_root: str = "thesis_results") -> None:
    output_dir = os.path.join(output_root, algo_name)
    os.makedirs(output_dir, exist_ok=True)
    if not reg_predictions:
        print(f"{algo_name}: no results to save.")
        return

    summary_table = pd.DataFrame([res["metrics"] for res in reg_predictions])
    summary_table.index = [f"Round_{i + 1}" for i in range(len(reg_predictions))]
    all_daily_df = pd.concat([res["daily_df"] for res in reg_predictions], keys=summary_table.index, names=["round", "date"])
    all_coefs = pd.concat([res["coef"] for res in reg_predictions], axis=1).T
    all_coefs.index = summary_table.index

    excel_path = os.path.join(output_dir, f"{algo_name}_Full_Results.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        summary_table.to_excel(writer, sheet_name="Performance_Summary")
        all_daily_df.to_excel(writer, sheet_name="Daily_Returns")
        all_coefs.to_excel(writer, sheet_name="Factor_Weights")
    print(f"數據已存至: {excel_path}")
    save_all_plots(all_daily_df.reset_index(level=0), all_coefs, algo_name, output_dir)


def _ensure_dir(folder: str) -> str:
    os.makedirs(folder, exist_ok=True)
    return folder


def _prepare_daily_df(daily_df: pd.DataFrame, date_col: str | None = None, ret_col: str = "strategy_ret") -> pd.DataFrame:
    df = daily_df.copy()
    if "round" in df.columns and date_col is None:
        # Keep all rounds, but use actual date index for charts.
        pass
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    else:
        for cand in ["年月日", "date", "Date", "交易日期"]:
            if cand in df.columns:
                df[cand] = pd.to_datetime(df[cand])
                df = df.set_index(cand)
                break
        if not isinstance(df.index, pd.DatetimeIndex):
            # If MultiIndex after save_algorithm_results, last level is often date.
            if isinstance(df.index, pd.MultiIndex):
                df.index = pd.to_datetime(df.index.get_level_values(-1))
            else:
                df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df[ret_col] = pd.to_numeric(df[ret_col], errors="coerce")
    df = df.dropna(subset=[ret_col])
    return df


def plot_equity_curve_thesis(daily_df, folder, algo_name="Model", ret_col="strategy_ret", date_col=None):
    _ensure_dir(folder)
    df = _prepare_daily_df(daily_df, date_col=date_col, ret_col=ret_col)
    equity = (1 + df[ret_col]).cumprod()
    if len(equity) > 0:
        start_date = equity.index[0] - pd.Timedelta(days=1)
        equity.loc[start_date] = 1.0
        equity = equity.sort_index()
    plt.figure(figsize=(12, 6))
    plt.plot(equity.index, equity.values, linewidth=2)
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.title(f"{algo_name} Cumulative Net Value")
    plt.xlabel("Date")
    plt.ylabel("Net Value")
    plt.tight_layout()
    path = os.path.join(folder, f"{algo_name}_equity_curve.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_drawdown_thesis(daily_df, folder, algo_name="Model", ret_col="strategy_ret", date_col=None):
    _ensure_dir(folder)
    df = _prepare_daily_df(daily_df, date_col=date_col, ret_col=ret_col)
    equity = (1 + df[ret_col]).cumprod()
    drawdown = equity / equity.cummax() - 1
    plt.figure(figsize=(12, 5))
    plt.plot(drawdown.index, drawdown.values, linewidth=1.8)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title(f"{algo_name} Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    path = os.path.join(folder, f"{algo_name}_drawdown.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_monthly_return_heatmap_thesis(daily_df, folder, algo_name="Model", ret_col="strategy_ret", date_col=None):
    if sns is None:
        return None
    _ensure_dir(folder)
    df = _prepare_daily_df(daily_df, date_col=date_col, ret_col=ret_col)
    monthly_ret = df[ret_col].resample("ME").apply(lambda x: (1 + x).prod() - 1)
    tmp = monthly_ret.to_frame("ret")
    tmp["Year"] = tmp.index.year
    tmp["Month"] = tmp.index.month
    heatmap_df = tmp.pivot(index="Year", columns="Month", values="ret")
    plt.figure(figsize=(10, 4.8))
    sns.heatmap(heatmap_df, annot=True, fmt=".2%", center=0)
    plt.title(f"{algo_name} Monthly Return Heatmap")
    plt.xlabel("Month")
    plt.ylabel("Year")
    plt.tight_layout()
    path = os.path.join(folder, f"{algo_name}_monthly_return_heatmap.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_return_distribution_thesis(daily_df, folder, algo_name="Model", ret_col="strategy_ret", date_col=None):
    _ensure_dir(folder)
    df = _prepare_daily_df(daily_df, date_col=date_col, ret_col=ret_col)
    values = df[ret_col] * 100
    plt.figure(figsize=(10, 5))
    plt.hist(values.dropna(), bins=50)
    plt.axvline(values.mean(), linestyle="--", linewidth=1.5, label=f"Mean = {values.mean():.3f}%")
    plt.title(f"{algo_name} Daily Return Distribution")
    plt.xlabel("Daily Return (%)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(folder, f"{algo_name}_return_distribution.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_rolling_sharpe_thesis(daily_df, folder, algo_name="Model", ret_col="strategy_ret", date_col=None, window=60, annualization=252):
    _ensure_dir(folder)
    df = _prepare_daily_df(daily_df, date_col=date_col, ret_col=ret_col)
    rolling = df[ret_col].rolling(window).mean() / df[ret_col].rolling(window).std() * np.sqrt(annualization)
    plt.figure(figsize=(12, 5))
    plt.plot(rolling.index, rolling.values, linewidth=1.8)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title(f"{algo_name} {window}-Day Rolling Sharpe")
    plt.xlabel("Date")
    plt.ylabel("Rolling Sharpe")
    plt.tight_layout()
    path = os.path.join(folder, f"{algo_name}_rolling_sharpe_{window}d.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_rolling_rank_ic_thesis(daily_df, folder, algo_name="Model", ic_col="rank_ic", date_col=None, window=60):
    _ensure_dir(folder)
    df = _prepare_daily_df(daily_df, date_col=date_col, ret_col="strategy_ret")
    if ic_col not in df.columns:
        return None
    rolling_ic = pd.to_numeric(df[ic_col], errors="coerce").rolling(window).mean()
    plt.figure(figsize=(12, 5))
    plt.plot(rolling_ic.index, rolling_ic.values, linewidth=1.8)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title(f"{algo_name} {window}-Day Rolling Rank IC")
    plt.xlabel("Date")
    plt.ylabel("Rolling Rank IC")
    plt.tight_layout()
    path = os.path.join(folder, f"{algo_name}_rolling_rank_ic_{window}d.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_feature_importance_thesis(coef_df, folder, algo_name="Model", top_n=20):
    _ensure_dir(folder)
    if coef_df is None or len(coef_df) == 0:
        return None
    if isinstance(coef_df, pd.Series):
        importance = coef_df.copy()
    else:
        numeric_coef = coef_df.select_dtypes(include=["number"]).copy()
        importance = numeric_coef.drop(columns=["const"], errors="ignore").abs().mean(axis=0)
    importance = importance.sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(10, max(5, 0.35 * len(importance))))
    importance.sort_values().plot(kind="barh")
    plt.title(f"{algo_name} Feature Importance")
    plt.xlabel("Mean Absolute Weight / Importance")
    plt.tight_layout()
    path = os.path.join(folder, f"{algo_name}_feature_importance.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def save_thesis_plots(daily_df, coef_df=None, algo_name="Model", folder="thesis_results/plots") -> dict:
    paths = {
        "equity_curve": plot_equity_curve_thesis(daily_df, folder, algo_name),
        "drawdown": plot_drawdown_thesis(daily_df, folder, algo_name),
        "monthly_heatmap": plot_monthly_return_heatmap_thesis(daily_df, folder, algo_name),
        "return_distribution": plot_return_distribution_thesis(daily_df, folder, algo_name),
        "rolling_sharpe": plot_rolling_sharpe_thesis(daily_df, folder, algo_name),
        "rolling_rank_ic": plot_rolling_rank_ic_thesis(daily_df, folder, algo_name),
        "feature_importance": plot_feature_importance_thesis(coef_df, folder, algo_name),
    }
    return {k: v for k, v in paths.items() if v is not None}


def save_all_plots(daily_df, coef_df, algo_name, folder):
    paths = save_thesis_plots(daily_df, coef_df, algo_name, folder)
    print(f"圖表已存入: {folder}/")
    for name, path in paths.items():
        print(f"  - {name}: {path}")
    return paths


def lstm_preprocessing(X: pd.DataFrame, exclude_cols: list[str] | None = None) -> pd.DataFrame:
    X_copy = X.copy()
    X_copy = X_copy.loc[:, ~X_copy.columns.duplicated()]
    default_exclude = ["年份", "報酬率%", "報酬率％", "代號", "名稱", "target_ret"]
    features = _feature_columns(X_copy, exclude_cols=(exclude_cols or default_exclude))
    if "年月日" not in X_copy.columns or "代號" not in X_copy.columns:
        raise KeyError("lstm_preprocessing requires columns '年月日' and '代號'.")
    X_copy[features] = X_copy[features].apply(pd.to_numeric, errors="coerce")
    X_copy[features] = X_copy.groupby("年月日")[features].transform(lambda s: s.fillna(s.mean()))
    standardized = X_copy.groupby("年月日")[features].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-8))
    X_cs = X_copy.copy()
    X_cs[features] = standardized
    X_cs = X_cs.sort_values(["代號", "年月日"])
    X_cs[features] = X_cs.groupby("代號")[features].ffill().bfill()
    X_cs[features] = X_cs[features].fillna(0)
    return X_cs


if nn is not None:
    class LSTMAutoencoder(nn.Module):
        def __init__(self, seq_len, n_features, embedding_dim=10):
            super().__init__()
            self.seq_len = seq_len
            self.n_features = n_features
            self.embedding_dim = embedding_dim
            self.encoder_lstm = nn.LSTM(n_features, embedding_dim, batch_first=True)
            self.decoder_lstm = nn.LSTM(embedding_dim, n_features, batch_first=True)
            self.decoder_output = nn.Linear(n_features, n_features)

        def forward(self, x):
            _, (hidden, _) = self.encoder_lstm(x)
            encoded = hidden.squeeze(0)
            decoded_inputs = encoded.unsqueeze(1).repeat(1, self.seq_len, 1)
            decoded, _ = self.decoder_lstm(decoded_inputs)
            decoded = self.decoder_output(decoded)
            return decoded, encoded
else:
    class LSTMAutoencoder:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for LSTMAutoencoder.")


def create_lstm_sequences(df: pd.DataFrame, features_cols: list[str], time_steps: int = 5, predict_horizon: int = 1):
    X_list, feature_index_list, target_index_list = [], [], []
    df = df.sort_values(["代號", "年月日"]).copy()
    features_cols = [c for c in features_cols if c in df.columns]
    df[features_cols] = df[features_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    for _, group in df.groupby("代號", sort=False):
        group_values = group[features_cols].values.astype(np.float32)
        group_indices = group.index.values
        max_i = len(group) - time_steps - predict_horizon + 1
        for i in range(max(0, max_i)):
            X_list.append(group_values[i : i + time_steps])
            feature_index_list.append(group_indices[i + time_steps - 1])
            target_index_list.append(group_indices[i + time_steps - 1 + predict_horizon])
    if not X_list:
        return np.empty((0, time_steps, len(features_cols)), dtype=np.float32), feature_index_list, target_index_list
    return np.array(X_list, dtype=np.float32), feature_index_list, target_index_list


def plot_loss_curve(train_history, val_history, round_name, patience_triggered=None, output_dir="loss_curves"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(train_history, label="Train Loss", linewidth=2)
    plt.plot(val_history, label="Validation Loss", linewidth=2)
    if patience_triggered is not None and len(train_history) > 0:
        best_epoch = int(np.nanargmin(val_history)) if len(val_history) else len(train_history) - 1
        plt.axvline(x=best_epoch, linestyle="--", label=f"Best Model (Epoch {best_epoch + 1})")
        plt.scatter(best_epoch, val_history[best_epoch], zorder=5)
    plt.title(f"LSTM Autoencoder Training Loss - {round_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.grid(True, alpha=0.3)
    path = os.path.join(output_dir, f"Loss_Curve_{round_name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def train_and_extract_pytorch(X_train, X_val, X_test=None, round_name="Round_X", encoding_dim=10, epochs=50, batch_size=128, patience=5, return_model_only=False):
    if torch is None:
        raise ImportError("PyTorch is required to train the LSTM autoencoder.")
    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("X_train and X_val must not be empty.")

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"[DEVICE] using {device}")
    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    val_tensor = torch.tensor(X_val, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(train_tensor, train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor, val_tensor), batch_size=batch_size, shuffle=False)

    seq_len, n_features = X_train.shape[1], X_train.shape[2]
    model = LSTMAutoencoder(seq_len, n_features, encoding_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_history, val_history = [], []
    writer = SummaryWriter(log_dir=f"runs/LSTM_{round_name}") if SummaryWriter is not None else None
    best_val_loss = float("inf")
    best_model_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0

    print(f"啟動訓練 ({round_name}), device={device}")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            decoded, _ = model(batch_x)
            loss = criterion(decoded, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                decoded, _ = model(batch_x)
                loss = criterion(decoded, batch_y)
                val_loss += loss.item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)
        train_history.append(train_loss)
        val_history.append(val_loss)
        if writer is not None:
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Validation", val_loss, epoch)
        print(f"Epoch {epoch + 1:02d}/{epochs} - Train Loss: {train_loss:.5f} - Val Loss: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"連續 {patience} 次 Val Loss 未下降，提前停止。")
                break
    if writer is not None:
        writer.close()
    plot_loss_curve(train_history, val_history, round_name, patience_triggered=patience_counter)
    model.load_state_dict(best_model_weights)
    if return_model_only:
        return model
    return model, train_history, val_history


def load_and_extract_features(X_3d, seq_len, n_features, model_path, encoding_dim=10):
    if torch is None:
        raise ImportError("PyTorch is required to load and extract LSTM features.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAutoencoder(seq_len, n_features, encoding_dim).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    X_tensor = torch.tensor(X_3d, dtype=torch.float32).to(device)
    enc_list = []
    with torch.no_grad():
        for start in range(0, len(X_tensor), 4096):
            _, encoded = model(X_tensor[start : start + 4096])
            enc_list.append(encoded.cpu().numpy())
    if not enc_list:
        return np.empty((0, encoding_dim), dtype=np.float32)
    return np.vstack(enc_list)
