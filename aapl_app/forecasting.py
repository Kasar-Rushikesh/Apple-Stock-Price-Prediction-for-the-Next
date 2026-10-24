"""
Core forecasting utilities for the Apple (AAPL) 30-day stock price project.

Models implemented
------------------
* Naive        : last observed close carried forward (baseline)
* ARIMA        : on log-prices, order picked by AIC from a small grid
* SARIMA       : ARIMA + weekly (5 trading-day) seasonality
* Random Forest: direct multi-horizon model on engineered features
* XGBoost      : direct multi-horizon model on engineered features

All ML models are *direct* multi-horizon models: a single model is trained on
stacked rows (features at day t, horizon h) -> log(Close[t+h] / Close[t]),
which avoids the error build-up of recursive forecasting.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

HORIZON = 30
MODEL_NAMES = ["Naive", "ARIMA", "SARIMA", "Random Forest", "XGBoost"]
DATA_PATH = Path(__file__).parent / "Dataset.csv"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_data(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.drop_duplicates("Date").sort_values("Date").set_index("Date")
    df = df.ffill().dropna()  # missing-value handling (dataset has none)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Moving averages, volatility, RSI, returns - used for EDA and ML."""
    out = df.copy()
    c = out["Close"]
    out["Return"] = c.pct_change()
    out["LogRet"] = np.log(c).diff()
    for w in (5, 10, 20, 50, 200):
        out[f"MA{w}"] = c.rolling(w).mean()
    out["Vol20"] = out["LogRet"].rolling(20).std() * np.sqrt(252)  # annualised
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    out["RSI14"] = 100 - 100 / (1 + gain / loss)
    out["Drawdown"] = c / c.cummax() - 1
    return out


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scale-free features (so the model works across price regimes)."""
    c, v = df["Close"], df["Volume"]
    lr = np.log(c).diff()
    f = pd.DataFrame(index=df.index)
    for k in (1, 2, 3, 5, 10, 20):
        f[f"ret_{k}"] = np.log(c).diff(k)
    for w in (5, 10, 20, 50):
        f[f"ma_ratio_{w}"] = c / c.rolling(w).mean() - 1
    f["vol_10"] = lr.rolling(10).std()
    f["vol_20"] = lr.rolling(20).std()
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi"] = 100 - 100 / (1 + gain / loss)
    f["vol_ratio"] = v / v.rolling(20).mean() - 1
    f["hl_range"] = (df["High"] - df["Low"]) / c
    f["dow"] = df.index.dayofweek
    f["month"] = df.index.month
    return f


def future_dates(last_date, horizon=HORIZON):
    """Next `horizon` business days after last_date."""
    return pd.bdate_range(pd.Timestamp(last_date) + pd.Timedelta(days=1), periods=horizon)


# --------------------------------------------------------------------------
# Statistical models
# --------------------------------------------------------------------------
def select_arima_order(log_close: pd.Series, max_p=3, max_q=3):
    best, best_aic = (1, 1, 1), np.inf
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                aic = ARIMA(log_close, order=(p, 1, q), trend="t").fit().aic
            except Exception:
                continue
            if aic < best_aic:
                best, best_aic = (p, 1, q), aic
    return best


def forecast_arima(close: pd.Series, order=None, horizon=HORIZON):
    y = np.log(close.values)
    order = order or select_arima_order(pd.Series(y))
    res = ARIMA(y, order=order, trend="t").fit()
    fc = res.get_forecast(horizon)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)
    return np.exp(mean), np.exp(ci[:, 0]), np.exp(ci[:, 1]), order


def forecast_sarima(close: pd.Series, order=(1, 1, 1), seasonal=(1, 0, 1, 5), horizon=HORIZON):
    y = np.log(close.values[-750:])  # last ~3 years keeps the fit fast & relevant
    res = SARIMAX(y, order=order, seasonal_order=seasonal, trend="t",
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=50)
    fc = res.get_forecast(horizon)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)
    return np.exp(mean), np.exp(ci[:, 0]), np.exp(ci[:, 1]), (order, seasonal)


# --------------------------------------------------------------------------
# Machine-learning models (direct multi-horizon)
# --------------------------------------------------------------------------
def _stack(df: pd.DataFrame, horizon=HORIZON):
    feats = make_features(df)
    logc = np.log(df["Close"])
    X_parts, y_parts = [], []
    for h in range(1, horizon + 1):
        target = logc.shift(-h) - logc
        part = feats.copy()
        part["h"] = h
        part["target"] = target
        X_parts.append(part)
    full = pd.concat(X_parts).dropna()
    return full.drop(columns="target"), full["target"]


def _make_model(kind):
    if kind == "XGBoost":
        return XGBRegressor(n_estimators=250, max_depth=3, learning_rate=0.03,
                            subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                            reg_lambda=5.0, n_jobs=2, random_state=42, verbosity=0)
    return RandomForestRegressor(n_estimators=120, max_depth=6, min_samples_leaf=50,
                                 max_features=0.6, n_jobs=2, random_state=42)


def forecast_ml(df: pd.DataFrame, kind="XGBoost", horizon=HORIZON):
    """Train on `df` (history up to the as-of date) and forecast `horizon` days."""
    X, y = _stack(df, horizon)
    model = _make_model(kind).fit(X, y)
    last = make_features(df).iloc[[-1]]
    rows = pd.concat([last.assign(h=h) for h in range(1, horizon + 1)])[X.columns]
    log_ret = model.predict(rows)
    price = df["Close"].iloc[-1] * np.exp(log_ret)

    # empirical residual spread -> approximate 95% band that widens with horizon
    resid_sd = np.std(y - model.predict(X))
    sd_h = resid_sd * np.sqrt(np.arange(1, horizon + 1) / horizon) * 1.0
    sd_h = np.maximum(sd_h, df["Close"].pipe(np.log).diff().std() * np.sqrt(np.arange(1, horizon + 1)) * 0.6)
    lo = price * np.exp(-1.96 * sd_h)
    hi = price * np.exp(1.96 * sd_h)
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return price, lo, hi, importances


def forecast_naive(close: pd.Series, horizon=HORIZON):
    last = close.iloc[-1]
    sd = np.log(close).diff().dropna().iloc[-250:].std()
    steps = np.arange(1, horizon + 1)
    price = np.repeat(last, horizon)
    return price, last * np.exp(-1.96 * sd * np.sqrt(steps)), last * np.exp(1.96 * sd * np.sqrt(steps))


def run_forecast(df: pd.DataFrame, model: str, horizon=HORIZON):
    """Unified entry point. Returns dict(price, lo, hi, info)."""
    close = df["Close"]
    info = {}
    if model == "Naive":
        p, lo, hi = forecast_naive(close, horizon)
    elif model == "ARIMA":
        p, lo, hi, order = forecast_arima(close, horizon=horizon)
        info["order"] = order
    elif model == "SARIMA":
        p, lo, hi, order = forecast_sarima(close, horizon=horizon)
        info["order"] = order
    else:
        p, lo, hi, imp = forecast_ml(df, model, horizon)
        info["importances"] = imp
    return dict(price=np.asarray(p), lo=np.asarray(lo), hi=np.asarray(hi), info=info)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def metrics(actual, pred):
    actual, pred = np.asarray(actual), np.asarray(pred)
    err = pred - actual
    return dict(
        MAE=float(np.mean(np.abs(err))),
        RMSE=float(np.sqrt(np.mean(err ** 2))),
        MAPE=float(np.mean(np.abs(err / actual)) * 100),
    )


def backtest(df: pd.DataFrame, n_origins=10, step=30, horizon=HORIZON, models=MODEL_NAMES):
    """Rolling-origin evaluation: train up to origin, forecast next 30 trading days,
    compare with actuals. Returns (summary_df, per_origin_df)."""
    n = len(df)
    origins = [n - horizon - step * i for i in range(n_origins)][::-1]
    rows = []
    for o in origins:
        train = df.iloc[:o]
        actual = df["Close"].iloc[o:o + horizon].values
        for m in models:
            res = run_forecast(train, m, horizon)
            mt = metrics(actual, res["price"])
            rows.append(dict(model=m, origin=train.index[-1], **mt,
                             direction_ok=bool(np.sign(res["price"][-1] - train["Close"].iloc[-1])
                                               == np.sign(actual[-1] - train["Close"].iloc[-1]))))
    per = pd.DataFrame(rows)
    summary = (per.groupby("model")
                  .agg(MAE=("MAE", "mean"), RMSE=("RMSE", "mean"), MAPE=("MAPE", "mean"),
                       DirectionalAccuracy=("direction_ok", "mean"))
                  .reindex(models).reset_index())
    summary["DirectionalAccuracy"] *= 100
    return summary, per
