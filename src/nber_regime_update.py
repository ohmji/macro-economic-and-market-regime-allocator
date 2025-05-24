import pandas as pd
nber = pd.read_csv("data/raw/nber_regimes.csv", parse_dates=["Date"], index_col="Date")
last_value = nber["EconRegime"].iloc[-1]

# สร้างช่วงต่อ
last_date = nber.index[-1]
fill_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), end="2025-05-01", freq="MS")
extended = pd.DataFrame(index=fill_dates, data={"EconRegime": last_value})

nber = pd.concat([nber, extended])
nber.to_csv("data/nber_regimes_filled.csv")
