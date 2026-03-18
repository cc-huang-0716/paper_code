import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr
import quantstats as qs

# 長短期策略函數

def long_short(group, y_col='報酬率%'):
    n_stocks = len(group)
    if n_stocks < 10:
        return pd.Series({'strategy_ret': 0, 'long_ret': 0, 'short_ret': 0})
    
    n_top = max(1, int(n_stocks * 0.1))
    sorted_group = group.sort_values('y_pred', ascending=False)
    long_ret = sorted_group.head(n_top)[y_col].mean() / 100
    short_ret = sorted_group.tail(n_top)[y_col].mean() / 100
    
    return pd.Series({
        'strategy_ret': long_ret - short_ret, 
        'long_ret': long_ret,
        'short_ret': short_ret
    })


def multi_reg(train_df, test_df, features, y_col="報酬率%", date_col="年月日", annualization=252):

    # 訓練模型
    X_train = train_df[features]
    X_train = sm.add_constant(X_train)
    model = sm.OLS(train_df[y_col], X_train).fit()
    
    # 樣本外預測
    test_data = test_df.copy()
    X_test = test_df[features] 
    X_test = sm.add_constant(X_test, has_constant='add')
    test_data['y_pred'] = model.predict(X_test)

    daily_performance = test_data.groupby(date_col).apply(long_short, y_col="報酬率%")
    strategy_rets = daily_performance['strategy_ret'] / 100
    strat_cum = (1 + strategy_rets / 100).cumprod()
    std = daily_performance['strategy_ret'].std()
    sharpe = (daily_performance['strategy_ret'].mean() / std) * np.sqrt(annualization) if std > 1e-6 else 0
    
    ic, _ = pearsonr(test_data[y_col], test_data['y_pred'])
    rank_ic, _ = spearmanr(test_data[y_col], test_data['y_pred'])

    invest_metrics = {
                    "cum_return": strat_cum.iloc[-1] - 1 if not strat_cum.empty else 0,
                    "sharpe": sharpe,
                    "max_drawdown": ((strat_cum - strat_cum.cummax()) / strat_cum.cummax()).min(),
                    "win_rate": np.mean(daily_performance['strategy_ret'] > 0), 
                    "avg_ic": ic,
                    "avg_rank_ic": rank_ic
                    }

    return {
        "model": model,
        "daily_df": daily_performance,
        "metrics": invest_metrics,
        "coef": model.params,       
        "tvalues": model.tvalues,   
        "pvalues": model.pvalues,
        "test_data": test_data   
            }

# ridge
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr, spearmanr
import os

def run_ridge_strategy(train_df, test_df, features, algo_name="Ridge", alpha=1.0, y_col="報酬率%", date_col="年月日"):

    # 數據準備
    X_train = train_df[features]
    y_train = train_df[y_col]
    X_test = test_df[features]
    
    # 模型訓練與預測
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_train, y_train)
    
    test_data = test_df.copy()
    test_data['y_pred'] = model.predict(X_test)

    # 執行多空對沖策略
    daily_performance = test_data.groupby(date_col).apply(long_short, y_col=y_col)
    
    # 指標計算
    strategy_rets = daily_performance['strategy_ret']
    cum_curve = (1 + strategy_rets / 100).cumprod()
    
    # IC / Rank IC
    ic, _ = pearsonr(test_data[y_col], test_data['y_pred'])
    rank_ic, _ = spearmanr(test_data[y_col], test_data['y_pred'])

    invest_metrics = {
        "algo": algo_name,
        "alpha": alpha,
        "cum_return": cum_curve.iloc[-1] - 1 if not cum_curve.empty else 0,
        "sharpe": (strategy_rets.mean() / strategy_rets.std()) * np.sqrt(252) if strategy_rets.std() != 0 else 0,
        "max_drawdown": ((cum_curve - cum_curve.cummax()) / cum_curve.cummax()).min() if not cum_curve.empty else 0,
        "rank_ic": rank_ic,
        "win_rate": (strategy_rets > 0).mean()
    }

    # Beta Coef Series
    coef_series = pd.Series(model.coef_, index=features)

    return {
        "metrics": invest_metrics,
        "daily_df": daily_performance,
        "coef": coef_series,
        "model": model,
        "test_data": test_data
    }

# lasso
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso
from scipy.stats import pearsonr, spearmanr
import os

def run_lasso_strategy(train_df, test_df, features, algo_name="Lasso", alpha=0.001, y_col="報酬率%", date_col="年月日"):

    X_train = train_df[features]
    y_train = train_df[y_col]
    X_test = test_df[features]
    
    # 訓練 Lasso 模型
    model = Lasso(alpha=alpha, fit_intercept=True, max_iter=10000)
    model.fit(X_train, y_train)
    
    test_data = test_df.copy()
    test_data['y_pred'] = model.predict(X_test)

    # 執行多空對沖策略
    daily_performance = test_data.groupby(date_col).apply(long_short, y_col=y_col)
    
    # 指標計算 (Sharpe, MDD, Rank IC 等)
    strategy_rets = daily_performance['strategy_ret']
    cum_curve = (1 + strategy_rets / 100).cumprod()
    rank_ic, _ = spearmanr(test_data[y_col], test_data['y_pred'])

    invest_metrics = {
        "algo": algo_name,
        "alpha": alpha,
        "cum_return": cum_curve.iloc[-1] - 1 if not cum_curve.empty else 0,
        "sharpe": (strategy_rets.mean() / strategy_rets.std()) * np.sqrt(252) if strategy_rets.std() != 0 else 0,
        "max_drawdown": ((cum_curve - cum_curve.cummax()) / cum_curve.cummax()).min() if not cum_curve.empty else 0,
        "rank_ic": rank_ic,
        "win_rate": (strategy_rets > 0).mean()
    }

    return {
        "metrics": invest_metrics,
        "daily_df": daily_performance,
        "coef": pd.Series(model.coef_, index=features),
        "model": model,
        "test_data": test_data
    }

#xgboost
import xgboost as xgb
from scipy.stats import spearmanr
import numpy as np
import pandas as pd

def run_xgboost_strategy(train_df, test_df, features, algo_name="XGBoost", y_col="報酬率%", date_col="年月日"):

    X_train = train_df[features]
    y_train = train_df[y_col]
    X_test = test_df[features]
    
    # 建立模型
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    # 訓練模型
    model.fit(X_train, y_train)
    
    # 樣本外預測
    test_data = test_df.copy()
    test_data['y_pred'] = model.predict(X_test)

    # 執行多空對沖策略
    daily_performance = test_data.groupby(date_col).apply(long_short, y_col=y_col)
    
    # 指標計算
    strategy_rets = daily_performance['strategy_ret']
    cum_curve = (1 + strategy_rets / 100).cumprod()
    rank_ic, _ = spearmanr(test_data[y_col], test_data['y_pred'])

    invest_metrics = {
        "algo": algo_name,
        "cum_return": cum_curve.iloc[-1] - 1 if not cum_curve.empty else 0,
        "sharpe": (strategy_rets.mean() / strategy_rets.std()) * np.sqrt(252) if strategy_rets.std() != 0 else 0,
        "max_drawdown": ((cum_curve - cum_curve.cummax()) / cum_curve.cummax()).min() if not cum_curve.empty else 0,
        "rank_ic": rank_ic,
        "win_rate": (strategy_rets > 0).mean()
    }

    # Feature Importance
    importance = pd.Series(model.feature_importances_, index=features)

    return {
        "metrics": invest_metrics,
        "daily_df": daily_performance,
        "coef": importance,
        "model": model,
        "test_data": test_data
    }

# random forest
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
import numpy as np
import pandas as pd

def run_rf_strategy(train_df, test_df, features, algo_name="RandomForest", y_col="報酬率%", date_col="年月日"):

    X_train = train_df[features]
    y_train = train_df[y_col]
    X_test = test_df[features]
    
    # 建立模型
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        max_features='sqrt',
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    # 訓練模型
    model.fit(X_train, y_train)
    
    # 樣本外預測
    test_data = test_df.copy()
    test_data['y_pred'] = model.predict(X_test)

    # 執行多空對沖策略
    daily_performance = test_data.groupby(date_col).apply(long_short, y_col=y_col)
    
    # 指標計算
    strategy_rets = daily_performance['strategy_ret']
    cum_curve = (1 + strategy_rets / 100).cumprod()
    rank_ic, _ = spearmanr(test_data[y_col], test_data['y_pred'])

    invest_metrics = {
        "algo": algo_name,
        "cum_return": cum_curve.iloc[-1] - 1 if not cum_curve.empty else 0,
        "sharpe": (strategy_rets.mean() / strategy_rets.std()) * np.sqrt(252) if strategy_rets.std() != 0 else 0,
        "max_drawdown": ((cum_curve - cum_curve.cummax()) / cum_curve.cummax()).min() if not cum_curve.empty else 0,
        "rank_ic": rank_ic,
        "win_rate": (strategy_rets > 0).mean()
    }

    # 特徵重要性
    importance = pd.Series(model.feature_importances_, index=features)

    return {
        "metrics": invest_metrics,
        "daily_df": daily_performance,
        "coef": importance,
        "model": model,
        "test_data": test_data
    }