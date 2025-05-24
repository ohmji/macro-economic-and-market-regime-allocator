import yfinance as yf
import pandas as pd

# Define the ticker symbol and date range
ticker = "^GSPC"  # S&P 500 Index
start_date = "2023-08-01"
end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

# Download monthly data
data = yf.download(ticker, start=start_date, end=end_date, interval="1mo")

# Clean and format the data
data = data[["Open", "High", "Low", "Close"]]
data.index = data.index.to_period("M").to_timestamp()
data.index.name = "Date"
data.reset_index(inplace=True)

# Export or print as CSV-format string
print(data.to_csv(index=False))