"""Apple (AAPL) Stock Price Forecast - next 30 days. Run: streamlit run app.py"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import forecasting as fc

st.set_page_config(page_title="AAPL 30-Day Forecast", page_icon="📈", layout="wide")


@st.cache_data
def get_data():
    return fc.add_indicators(fc.load_data())


@st.cache_data(show_spinner=False)
def get_forecast(asof: str, model: str):
    df = fc.load_data().loc[:asof]
    return fc.run_forecast(df, model)


df = get_data()
raw = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]

# ---------------------------------------------------------------- sidebar
st.sidebar.title("⚙️ Controls")
min_d, max_d = df.index.min().date(), df.index.max().date()
asof = st.sidebar.date_input(
    "Forecast from (as-of date)", value=max_d,
    min_value=(df.index.min() + pd.Timedelta(days=400)).date(), max_value=max_d,
    help="The model only sees data up to this date and predicts the next 30 trading days.")
model_name = st.sidebar.selectbox("Model", fc.MODEL_NAMES, index=2)
history_days = st.sidebar.slider("History shown (trading days)", 60, 750, 250, 10)
show_ci = st.sidebar.checkbox("Show 95% confidence band", True)
st.sidebar.caption("If the as-of date is before the end of the data, actual prices are overlaid so you can see forecast error.")

# ---------------------------------------------------------------- header
st.title("📈 Apple (AAPL) Stock Price Forecast – Next 30 Days")
st.caption(f"Daily data {min_d:%d %b %Y} – {max_d:%d %b %Y} · {len(df):,} trading days · "
           "Models: ARIMA, SARIMA, Random Forest, XGBoost")

tab_fc, tab_eda, tab_cmp, tab_about = st.tabs(
    ["🔮 Forecast", "📊 EDA & Trends", "🏆 Model Comparison", "ℹ️ About"])

# ---------------------------------------------------------------- forecast
with tab_fc:
    asof_ts = pd.Timestamp(asof)
    train_idx = df.index[df.index <= asof_ts]
    asof_actual = train_idx[-1]
    with st.spinner(f"Fitting {model_name}…"):
        res = get_forecast(str(asof_actual.date()), model_name)
    dates = fc.future_dates(asof_actual)
    last_close = float(df.loc[asof_actual, "Close"])
    end_price = float(res["price"][-1])
    chg = end_price / last_close - 1

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Last close ({asof_actual:%d %b %Y})", f"${last_close:,.2f}")
    c2.metric(f"Forecast {dates[-1]:%d %b %Y}", f"${end_price:,.2f}", f"{chg:+.1%}")
    c3.metric("Forecast high / low", f"${res['price'].max():,.2f} / ${res['price'].min():,.2f}")
    c4.metric("95% range on day 30", f"${res['lo'][-1]:,.0f} – ${res['hi'][-1]:,.0f}")

    hist = df.loc[:asof_actual].iloc[-history_days:]
    fig = go.Figure()
    fig.add_scatter(x=hist.index, y=hist["Close"], name="Historical close", line=dict(color="#1f77b4"))
    if show_ci:
        fig.add_scatter(x=list(dates) + list(dates[::-1]),
                        y=list(res["hi"]) + list(res["lo"][::-1]), fill="toself",
                        fillcolor="rgba(255,127,14,0.18)", line=dict(width=0),
                        name="95% band", hoverinfo="skip")
    fig.add_scatter(x=dates, y=res["price"], name=f"{model_name} forecast",
                    line=dict(color="#ff7f0e", dash="dash", width=3))
    # overlay actuals if available
    actual = df["Close"].loc[(df.index >= dates[0]) & (df.index <= dates[-1])]
    err_df = None
    if len(actual):
        fig.add_scatter(x=actual.index, y=actual, name="Actual", line=dict(color="#2ca02c", width=2))
        n = len(actual)
        m = fc.metrics(actual.values, res["price"][:n])
        err_df = m
    fig.update_layout(height=520, hovermode="x unified", yaxis_title="Price (USD)",
                      legend=dict(orientation="h", y=1.08), margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width='stretch')

    if err_df:
        st.success(f"Backtest on actual data ({len(actual)} days available) – "
                   f"MAE ${err_df['MAE']:.2f} · RMSE ${err_df['RMSE']:.2f} · MAPE {err_df['MAPE']:.2f}%")
    else:
        st.info("Forecast extends beyond the dataset, so there are no actual prices to compare against.")

    left, right = st.columns([3, 2])
    with left:
        table = pd.DataFrame({"Date": dates.date, "Forecast ($)": res["price"].round(2),
                              "Lower 95% ($)": res["lo"].round(2), "Upper 95% ($)": res["hi"].round(2)})
        st.subheader("Forecast table")
        st.dataframe(table, width='stretch', hide_index=True, height=300)
        st.download_button("⬇️ Download forecast (CSV)", table.to_csv(index=False).encode(),
                           f"aapl_forecast_{model_name.replace(' ', '_')}_{asof_actual:%Y%m%d}.csv", "text/csv")
    with right:
        imp = res["info"].get("importances")
        if imp is not None:
            st.subheader("Top features")
            top = imp.head(8)[::-1]
            f2 = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h", marker_color="#ff7f0e"))
            f2.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(f2, width='stretch')
        elif "order" in res["info"]:
            st.subheader("Model specification")
            st.code(f"{model_name} order = {res['info']['order']}")
            st.caption("ARIMA order chosen by AIC grid search; SARIMA uses a 5-day (weekly) seasonal term.")

    st.caption("⚠️ Educational project – not financial advice. Stock prices are close to a random walk; "
               "short-term forecasts carry large uncertainty (see the confidence band).")

# ---------------------------------------------------------------- EDA
with tab_eda:
    st.subheader("Price, moving averages and volume")
    yr = st.slider("Year range", df.index.year.min(), df.index.year.max(),
                   (df.index.year.min(), df.index.year.max()))
    d = df[(df.index.year >= yr[0]) & (df.index.year <= yr[1])]
    f = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.03)
    f.add_trace(go.Candlestick(x=d.index, open=d.Open, high=d.High, low=d.Low, close=d.Close, name="OHLC"), 1, 1)
    for w, col in [(20, "#ff7f0e"), (50, "#2ca02c"), (200, "#d62728")]:
        f.add_scatter(x=d.index, y=d[f"MA{w}"], name=f"MA{w}", line=dict(color=col, width=1.4), row=1, col=1)
    f.add_bar(x=d.index, y=d.Volume, name="Volume", marker_color="#9aa5b1", row=2, col=1)
    f.update_layout(height=620, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10),
                    legend=dict(orientation="h", y=1.04))
    st.plotly_chart(f, width='stretch')

    a, b = st.columns(2)
    with a:
        st.subheader("Annualised volatility (20-day)")
        fv = go.Figure(go.Scatter(x=d.index, y=d.Vol20, line=dict(color="#9467bd")))
        fv.update_layout(height=300, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fv, width='stretch')
    with b:
        st.subheader("Average monthly return (seasonality)")
        mret = df["Close"].resample("ME").last().pct_change().dropna()
        seas = mret.groupby(mret.index.month).mean() * 100
        fm = go.Figure(go.Bar(x=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                              y=seas.values, marker_color=np.where(seas.values >= 0, "#2ca02c", "#d62728")))
        fm.update_layout(height=300, yaxis_title="%", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fm, width='stretch')

    a, b = st.columns(2)
    with a:
        st.subheader("Yearly summary")
        ys = df.groupby(df.index.year).agg(Close_end=("Close", "last"), Low=("Low", "min"), High=("High", "max"),
                                           AvgVolume=("Volume", "mean"))
        ys["Return %"] = df["Close"].groupby(df.index.year).apply(lambda s: (s.iloc[-1] / s.iloc[0] - 1) * 100)
        st.dataframe(ys.round(2), width='stretch')
    with b:
        st.subheader("Daily return distribution")
        fh = go.Figure(go.Histogram(x=df["Return"].dropna() * 100, nbinsx=80, marker_color="#1f77b4"))
        fh.update_layout(height=300, xaxis_title="Daily return %", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fh, width='stretch')

    st.subheader("Key events visible in the data")
    ev = pd.DataFrame([
        ("Aug 2014", "7-for-1 stock split (June 2014) – prices in the dataset are split-adjusted"),
        ("Aug 2015", "China devaluation sell-off – sharp one-week drop"),
        ("Jan 2016", "Weak iPhone outlook / global growth fears – drawdown"),
        ("Nov 2016", "US election rally and iPhone 7 cycle"),
        ("Dec 2018", "Global market sell-off – largest drawdown in the sample"),
        ("2019", "Strong rebound; services growth and Fed rate cuts"),
    ], columns=["When", "Event"])
    st.dataframe(ev, hide_index=True, width='stretch')
    mdd = df["Drawdown"].min()
    st.caption(f"Maximum drawdown in the sample: {mdd:.1%} (on {df['Drawdown'].idxmin():%d %b %Y}).")

# ---------------------------------------------------------------- comparison
with tab_cmp:
    st.subheader("Rolling-origin backtest (8 origins × 30-day horizon, 2018–2019)")
    p = Path(__file__).parent / "backtest_summary.csv"
    if p.exists():
        s = pd.read_csv(p)
        best = s.loc[s["RMSE"].idxmin(), "model"]
        st.dataframe(s.round(2).style.highlight_min(subset=["MAE", "RMSE", "MAPE"], color="#c6efce"),
                     hide_index=True, width='stretch')
        fb = go.Figure(go.Bar(x=s["model"], y=s["RMSE"], marker_color=["#2ca02c" if m == best else "#9aa5b1" for m in s["model"]]))
        fb.update_layout(height=320, yaxis_title="RMSE ($)", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fb, width='stretch')
        st.markdown(f"""
**Best model by RMSE: {best}.**  
Each model is trained only on data before each origin, forecasts the next 30 trading days, and is scored against
what actually happened. Averaged over 8 origins.

**Honest reading of the results**
- Errors are large for every model (≈5% MAPE): daily stock prices behave close to a random walk.
- ARIMA/SARIMA do best mainly because they capture the **upward drift** of AAPL in 2018-19. Tree models predict
  mean-reverting returns and are more conservative.
- A forecast should be read together with its confidence band, not as a point guarantee.
""")
    else:
        st.warning("backtest_summary.csv not found – run `python train_evaluate.py` to generate it.")

# ---------------------------------------------------------------- about
with tab_about:
    st.markdown("""
### Project summary
Forecasts Apple's closing price for the next 30 trading days from daily OHLCV data (2012–2019).

**Pipeline** – cleaning → feature engineering (returns, MA ratios, volatility, RSI, volume ratio, calendar) →
EDA → ARIMA / SARIMA / Random Forest / XGBoost → rolling backtest → Streamlit deployment.

**Method notes**
- Statistical models work on log-prices; forecasts are exponentiated back to dollars.
- ML models are *direct multi-horizon* (one model, horizon as a feature, target = log-return to day *h*), avoiding recursive error build-up.
- Future dates are business days (US market holidays are not excluded).
- The dataset has no external series (S&P 500, inflation, earnings), so those are not used; adding them is the natural next step.

**Limitations** – the model cannot anticipate news or shocks; past performance ≠ future results. Not financial advice.
""")
