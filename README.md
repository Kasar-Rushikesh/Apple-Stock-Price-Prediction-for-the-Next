# 📈 Apple (AAPL) Stock Price Forecast – Next 30 Days

Streamlit app that forecasts Apple's closing price for the next 30 trading days using daily data from 2012–2019.

**Models:** Naive baseline, ARIMA, SARIMA, Random Forest, XGBoost
**App tabs:** Forecast (95% band, CSV download), EDA & Trends, Model Comparison, About

## Files
- `app.py` – Streamlit web app (entry point)
- `forecasting.py` – data prep, feature engineering, models, backtesting
- `Dataset.csv` – daily AAPL OHLCV data
- `backtest_summary.csv` – model evaluation results
- `requirements.txt` – dependencies

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Backtest results (8 rolling origins x 30-day horizon)
| Model | MAE ($) | RMSE ($) | MAPE (%) |
|---|---|---|---|
| Naive | 11.88 | 13.81 | 5.62 |
| ARIMA | 10.56 | 12.37 | 5.03 |
| **SARIMA** | **10.18** | **11.95** | **4.91** |
| Random Forest | 11.57 | 13.51 | 5.47 |
| XGBoost | 12.63 | 14.46 | 6.04 |

> Educational project, not financial advice.

<img width="945" height="437" alt="Screenshot 2026-10-24 124416" src="https://github.com/user-attachments/assets/4a337849-3bcb-440f-a05e-b71b61eef4e8" />
<img width="937" height="440" alt="Screenshot 2026-10-24 124439" src="https://github.com/user-attachments/assets/e93a83bb-3ea9-41e2-8206-2064c1c1e39f" />
<img width="954" height="507" alt="Screenshot 2026-10-24 124455" src="https://github.com/user-attachments/assets/d2165070-b997-42e6-8c0d-601ceb13ef20" />

