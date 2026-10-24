# 📈 Apple (AAPL) Stock Price Forecast – Next 30 Days

Streamlit app that forecasts Apple's closing price for the next 30 trading days using daily data from 2012–2019.

**Models:** Naive baseline · ARIMA · SARIMA · Random Forest · XGBoost
**App tabs:** Forecast (with 95% band, CSV download) · EDA & Trends · Model Comparison · About

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit web app (entry point) |
| `forecasting.py` | Data prep, feature engineering, models, backtesting |
| `train_evaluate.py` | Re-creates backtest results (~2 min) |
| `Dataset.csv` | Daily AAPL OHLCV data |
| `backtest_summary.csv`, `backtest_detail.csv` | Pre-computed model evaluation |
| `requirements.txt` | Dependencies |

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push all files in this folder to the **root** of a GitHub repo.
2. Go to https://share.streamlit.io → **New app** → pick the repo and branch.
3. Main file path: `app.py` → **Deploy**.

## Backtest results (8 rolling origins × 30-day horizon)
| Model | MAE ($) | RMSE ($) | MAPE (%) |
|---|---|---|---|
| Naive | 11.88 | 13.81 | 5.62 |
| ARIMA | 10.56 | 12.37 | 5.03 |
| **SARIMA** | **10.18** | **11.95** | **4.91** |
| Random Forest | 11.57 | 13.51 | 5.47 |
| XGBoost | 12.63 | 14.46 | 6.04 |

> Educational project, not financial advice. Stock prices are close to a random walk, so expect wide uncertainty.
