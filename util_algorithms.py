"""Utility functions for thesis prediction models and long-short backtests.

Key convention used in this fixed version:
- Original y_col is assumed to be percentage return, e.g. 1.25 means 1.25%.
- long_ret, short_ret and strategy_ret are stored as decimal returns, e.g. 0.0125 means 1.25%.
- cum_return, max_drawdown and Sharpe are therefore computed directly from decimal returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

try:
    import xgboost as xgb
except ImportError:  # Keep module importable even if xgboost is not installed yet.
    xgb = None


def _safe_corr(y_true: pd.Series, y_pred: pd.Series, method: str = "pearson") -> float:
    """Return correlation safely; return np.nan if values are constant or insufficient."""
    df = pd.DataFrame({"y": y_true, "pred": y_pred}).dropna()
    if len(df) < 3 or df["y"].nunique() <= 1 or df["pred"].nunique() <= 1:
        return np.nan
    if method == "spearman":
        return float(spearmanr(df["y"], df["pred"]).correlation)
    return float(pearsonr(df["y"], df["pred"])[0])


def long_short(group: pd.DataFrame, y_col: str = "報酬率%", pred_col: str = "y_pred", top_frac: float = 0.1) -> pd.Series:
    """Build a cross-sectional top-minus-bottom portfolio for one date.

    Returns are decimal returns. If y_col is in percent, it is divided by 100 once here.
    """
    group = group.dropna(subset=[y_col, pred_col]).copy()
    n_stocks = len(group)
    if n_stocks < 10:
        return pd.Series({"strategy_ret": 0.0, "long_ret": 0.0, "short_ret": 0.0, "n_stocks": n_stocks})

    n_top = max(1, int(n_stocks * top_frac))
    sorted_group = group.sort_values(pred_col, ascending=False)

    long_ret = sorted_group.head(n_top)[y_col].mean() / 100.0
    short_ret = sorted_group.tail(n_top)[y_col].mean() / 100.0

    return pd.Series({
        "strategy_ret": float(long_ret - short_ret),
        "long_ret": float(long_ret),
        "short_ret": float(short_ret),
        "n_stocks": int(n_stocks),
    })


def _daily_rank_ic(test_data: pd.DataFrame, y_col: str, date_col: str, pred_col: str = "y_pred") -> pd.Series:
    """Daily Spearman rank IC for rolling IC plots and summary."""
    def _one_day(g: pd.DataFrame) -> float:
        g = g[[y_col, pred_col]].dropna()
        if len(g) < 3 or g[y_col].nunique() <= 1 or g[pred_col].nunique() <= 1:
            return np.nan
        return float(spearmanr(g[y_col], g[pred_col]).correlation)

    return test_data.groupby(date_col, sort=True).apply(_one_day).rename("rank_ic")


def _calc_metrics(test_data: pd.DataFrame, daily_performance: pd.DataFrame, algo_name: str, y_col: str, date_col: str, annualization: int = 252, extra: dict | None = None) -> dict:
    """Calculate portfolio and prediction metrics from decimal daily returns."""
    strategy_rets = pd.to_numeric(daily_performance["strategy_ret"], errors="coerce").dropna()
    cum_curve = (1.0 + strategy_rets).cumprod()
    std = strategy_rets.std()

    ic = _safe_corr(test_data[y_col], test_data["y_pred"], method="pearson")
    rank_ic = _safe_corr(test_data[y_col], test_data["y_pred"], method="spearman")
    daily_ic = _daily_rank_ic(test_data, y_col=y_col, date_col=date_col)

    metrics = {
        "algo": algo_name,
        "n_test_rows": int(len(test_data)),
        "n_test_pairs": int(test_data[[date_col, "代號"]].drop_duplicates().shape[0]) if "代號" in test_data.columns else int(len(test_data)),
        "start_date": pd.to_datetime(test_data[date_col]).min() if date_col in test_data.columns else None,
        "end_date": pd.to_datetime(test_data[date_col]).max() if date_col in test_data.columns else None,
        "cum_return": float(cum_curve.iloc[-1] - 1.0) if not cum_curve.empty else 0.0,
        "annual_return": float((cum_curve.iloc[-1] ** (annualization / len(strategy_rets)) - 1.0)) if len(strategy_rets) > 0 and not cum_curve.empty and cum_curve.iloc[-1] > 0 else np.nan,
        "annual_vol": float(std * np.sqrt(annualization)) if pd.notna(std) else np.nan,
        "sharpe": float(strategy_rets.mean() / std * np.sqrt(annualization)) if pd.notna(std) and std > 1e-12 else 0.0,
        "max_drawdown": float((cum_curve / cum_curve.cummax() - 1.0).min()) if not cum_curve.empty else 0.0,
        "win_rate": float((strategy_rets > 0).mean()) if len(strategy_rets) else np.nan,
        "avg_ic": ic,
        "rank_ic": rank_ic,
        "avg_daily_rank_ic": float(daily_ic.mean()) if daily_ic.notna().any() else np.nan,
    }
    if extra:
        metrics.update(extra)
    return metrics


def _prepare_model_data(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str], y_col: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    X_train = train_df[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y_train = pd.to_numeric(train_df[y_col], errors="coerce")
    train_mask = y_train.notna()

    X_test = test_df[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X_train.loc[train_mask], y_train.loc[train_mask], X_test


def multi_reg(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str], y_col: str = "報酬率%", date_col: str = "年月日", annualization: int = 252) -> dict:
    """OLS regression plus cross-sectional long-short evaluation."""
    X_train, y_train, X_test_raw = _prepare_model_data(train_df, test_df, features, y_col)
    X_train_const = sm.add_constant(X_train, has_constant="add")
    model = sm.OLS(y_train, X_train_const).fit()

    test_data = test_df.copy()
    X_test_const = sm.add_constant(X_test_raw, has_constant="add")
    test_data["y_pred"] = model.predict(X_test_const)
    test_data = test_data.dropna(subset=[y_col, "y_pred", date_col]).copy()

    daily_performance = test_data.groupby(date_col, sort=True).apply(long_short, y_col=y_col, pred_col="y_pred")
    daily_ic = _daily_rank_ic(test_data, y_col=y_col, date_col=date_col)
    daily_performance = daily_performance.join(daily_ic, how="left")

    invest_metrics = _calc_metrics(test_data, daily_performance, "OLS", y_col, date_col, annualization)

    return {
        "model": model,
        "daily_df": daily_performance,
        "metrics": invest_metrics,
        "coef": model.params,
        "tvalues": model.tvalues,
        "pvalues": model.pvalues,
        "test_data": test_data,
    }


def run_ridge_strategy(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str], algo_name: str = "Ridge", alpha: float = 1.0, y_col: str = "報酬率%", date_col: str = "年月日") -> dict:
    X_train, y_train, X_test = _prepare_model_data(train_df, test_df, features, y_col)
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_train, y_train)

    test_data = test_df.copy()
    test_data["y_pred"] = model.predict(X_test)
    test_data = test_data.dropna(subset=[y_col, "y_pred", date_col]).copy()

    daily_performance = test_data.groupby(date_col, sort=True).apply(long_short, y_col=y_col, pred_col="y_pred")
    daily_performance = daily_performance.join(_daily_rank_ic(test_data, y_col=y_col, date_col=date_col), how="left")
    metrics = _calc_metrics(test_data, daily_performance, algo_name, y_col, date_col, extra={"alpha": alpha})

    return {"metrics": metrics, "daily_df": daily_performance, "coef": pd.Series(model.coef_, index=features), "model": model, "test_data": test_data}


def run_lasso_strategy(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str], algo_name: str = "Lasso", alpha: float = 0.001, y_col: str = "報酬率%", date_col: str = "年月日") -> dict:
    X_train, y_train, X_test = _prepare_model_data(train_df, test_df, features, y_col)
    model = Lasso(alpha=alpha, fit_intercept=True, max_iter=10000)
    model.fit(X_train, y_train)

    test_data = test_df.copy()
    test_data["y_pred"] = model.predict(X_test)
    test_data = test_data.dropna(subset=[y_col, "y_pred", date_col]).copy()

    daily_performance = test_data.groupby(date_col, sort=True).apply(long_short, y_col=y_col, pred_col="y_pred")
    daily_performance = daily_performance.join(_daily_rank_ic(test_data, y_col=y_col, date_col=date_col), how="left")
    metrics = _calc_metrics(test_data, daily_performance, algo_name, y_col, date_col, extra={"alpha": alpha})

    return {"metrics": metrics, "daily_df": daily_performance, "coef": pd.Series(model.coef_, index=features), "model": model, "test_data": test_data}


def run_xgboost_strategy(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str], algo_name: str = "XGBoost", y_col: str = "報酬率%", date_col: str = "年月日") -> dict:
    if xgb is None:
        raise ImportError("xgboost is not installed. Please install xgboost before running this model.")
    X_train, y_train, X_test = _prepare_model_data(train_df, test_df, features, y_col)
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train)

    test_data = test_df.copy()
    test_data["y_pred"] = model.predict(X_test)
    test_data = test_data.dropna(subset=[y_col, "y_pred", date_col]).copy()

    daily_performance = test_data.groupby(date_col, sort=True).apply(long_short, y_col=y_col, pred_col="y_pred")
    daily_performance = daily_performance.join(_daily_rank_ic(test_data, y_col=y_col, date_col=date_col), how="left")
    metrics = _calc_metrics(test_data, daily_performance, algo_name, y_col, date_col)
    importance = pd.Series(model.feature_importances_, index=features)

    return {"metrics": metrics, "daily_df": daily_performance, "coef": importance, "model": model, "test_data": test_data}


def run_rf_strategy(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str], algo_name: str = "RandomForest", y_col: str = "報酬率%", date_col: str = "年月日") -> dict:
    X_train, y_train, X_test = _prepare_model_data(train_df, test_df, features, y_col)
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        max_features="sqrt",
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    test_data = test_df.copy()
    test_data["y_pred"] = model.predict(X_test)
    test_data = test_data.dropna(subset=[y_col, "y_pred", date_col]).copy()

    daily_performance = test_data.groupby(date_col, sort=True).apply(long_short, y_col=y_col, pred_col="y_pred")
    daily_performance = daily_performance.join(_daily_rank_ic(test_data, y_col=y_col, date_col=date_col), how="left")
    metrics = _calc_metrics(test_data, daily_performance, algo_name, y_col, date_col)
    importance = pd.Series(model.feature_importances_, index=features)

    return {"metrics": metrics, "daily_df": daily_performance, "coef": importance, "model": model, "test_data": test_data}
