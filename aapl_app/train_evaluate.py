"""Re-generate backtest_summary.csv / backtest_detail.csv (takes ~2 minutes)."""
import forecasting as fc

df = fc.load_data()
summary, detail = fc.backtest(df, n_origins=8, step=30)
summary.loc[summary.model == "Naive", "DirectionalAccuracy"] = float("nan")
summary.to_csv("backtest_summary.csv", index=False)
detail.to_csv("backtest_detail.csv", index=False)
print(summary.round(2))
