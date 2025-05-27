import numpy as np
import pandas as pd
import yfinance as yf
import os

# Import matplotlib.pyplot at the top for deduplication
import matplotlib.pyplot as plt

# PortfolioBacktester class for backtesting portfolio strategies
class PortfolioBacktester:
    def __init__(self, returns, weights, initial_value=1_000_000, start_date=None, end_date=None):
        self.weights = weights
        self.returns = returns
        self.initial_value = initial_value
        self.start_date = start_date
        self.end_date = end_date
        self.daily_returns = None
        self.equity_curve = None

    def run(self):
        assert np.allclose(self.weights.sum(axis=1), 1.0), "Weights must sum to 1 across assets for each time period"
        assert (self.weights >= 0).values.all(), "Weights must be non-negative"
        self.daily_returns = (self.returns.values * self.weights.values).sum(axis=1)
        self.daily_returns = pd.Series(self.daily_returns, index=self.returns.index)
        equity_curve = pd.Series((1 + self.daily_returns).cumprod() * self.initial_value)
        drawdown = equity_curve / equity_curve.cummax() - 1
        self.drawdown = drawdown
        self.max_drawdown = drawdown.min()
        self.equity_curve = equity_curve
        downside_returns = self.daily_returns[self.daily_returns < 0]
        self.var_95 = -np.percentile(self.daily_returns, 5)
        self.cvar_95 = -downside_returns[downside_returns <= -self.var_95].mean()
        return self

    def summary(self, risk_free_rate=0.03):
        if self.equity_curve is None:
            raise ValueError("Equity curve not found. Please run backtest first.")
        total_return = self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1
        cagr = (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) ** (1 / (len(self.equity_curve) / 252)) - 1
        volatility = self.daily_returns.std() * np.sqrt(252)
        sharpe = (self.daily_returns.mean() * 252 - risk_free_rate) / volatility

        summary_dict = {
            'Total Return': total_return,
            'CAGR': cagr,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': self.max_drawdown,
            'CVaR (95%)': self.cvar_95
        }

        return summary_dict


# Helper function to save plots (extracted for reuse)
def save_plot(series, title, ylabel, filename, report_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(series)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(report_dir, filename)
    plt.savefig(path)
    plt.close()
    print(f"Saved {title.lower()} to: {path}")

def run_backtest_and_save_summary(returns_df, weights_df, label, report_dir, version_suffix=""):
    """
    Run backtest and save summary, weights, and equity curve files with an optional version suffix.
    """
    suffix = version_suffix if version_suffix else ""
    print(f"\nBacktest for {label}{suffix}")
    backtester = PortfolioBacktester(returns=returns_df, weights=weights_df)
    result = backtester.run()
    weights_out = weights_df.copy()
    weights_out.index.name = 'Date'
    weights_out.to_csv(os.path.join(report_dir, f'{label.lower().replace(" ", "_")}_weights{suffix}.csv'))
    summary = result.summary()
    summary_df = pd.DataFrame([summary])
    print(f"\n{label}{suffix} Summary:\n", summary_df.T.round(4))
    summary_path = os.path.join(report_dir, f'{label.lower().replace(" ", "_")}_summary{suffix}.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to: {summary_path}")
    # Save equity curve PNG as well
    try:
        save_plot(result.equity_curve, f"{label}{suffix} Equity Curve", "Portfolio Value",
                  f'{label.lower().replace(" ", "_")}_equity_curve{suffix}.png', report_dir)
    except Exception as e:
        print(f"Warning: Could not save equity curve plot for {label}{suffix}: {e}")
    # Save drawdown curve PNG
    try:
        save_plot(result.drawdown, f"{label}{suffix} Drawdown Curve", "Drawdown",
                  f'{label.lower().replace(" ", "_")}_drawdown_curve{suffix}.png', report_dir)
    except Exception as e:
        print(f"Warning: Could not save drawdown curve plot for {label}{suffix}: {e}")
    return result

def run_regime_backtest_from_file(regime_filename, returns_df, report_dir, label_prefix):
    """
    Load regime predictions from file, convert to weights, run backtest, and save results.
    Returns the backtest result and version suffix used.
    """
    from src.allocator import regime_series_to_weights

    regime_df = pd.read_csv(regime_filename, parse_dates=['Date'])
    regime_df.set_index('Date', inplace=True)
    regime_series = regime_df['Regime'].map({0.0: 'Expansion', 1.0: 'Recession'})

    regime_file_lower = regime_filename.lower()
    if "econ" in regime_file_lower:
        version_suffix = "_econ"
    elif "mkt" in regime_file_lower or "market" in regime_file_lower:
        version_suffix = "_market"
    else:
        version_suffix = ""

    weights_model = regime_series_to_weights(regime_series)

    weights_model = weights_model.reindex(returns_df.index).ffill()
    weights_model = weights_model[returns_df.columns]

    valid_rows = weights_model.sum(axis=1) > 0
    weights_model = weights_model[valid_rows]
    returns_df_filtered = returns_df[valid_rows]

    weights_model = weights_model.div(weights_model.sum(axis=1), axis=0)

    result = run_backtest_and_save_summary(returns_df_filtered, weights_model, label_prefix, report_dir, version_suffix=version_suffix)
    print(f"\n=== {label_prefix}{version_suffix} Summary ===")
    summary_df = pd.DataFrame([result.summary()])
    print(summary_df.round(4))
    summary_file = f"{label_prefix.lower().replace(' ', '_')}_summary{version_suffix}.csv"
    print(f"Saved summary to: {os.path.join(report_dir, summary_file)}")
    weights_file = f"{label_prefix.lower().replace(' ', '_')}_weights{version_suffix}.csv"
    print(f"Saved weights to: {os.path.join(report_dir, weights_file)}")
    return result, version_suffix

if __name__ == "__main__":

    # Load SPY monthly returns
    spy = yf.download('SPY', start='2000-01-01', interval='1mo')['Close']
    returns_df = spy.pct_change().dropna()
    returns_df.columns = ['SPY']

    # Use 100% weight on all assets for all periods (buy-and-hold strategy)
    weights_df = pd.DataFrame(1.0, index=returns_df.index, columns=returns_df.columns)

    # Set up report directory
    report_dir = './data/report'
    os.makedirs(report_dir, exist_ok=True)

    # Run backtest and save summary for buy-and-hold (always _econ)
    result = run_backtest_and_save_summary(returns_df, weights_df, "Buy-and-Hold SPY", report_dir, version_suffix="")
    print("\n=== Buy-and-Hold SPY Summary ===")
    summary_df = pd.DataFrame([result.summary()])
    print(summary_df.round(4))
    print(f"Saved summary to: {os.path.join(report_dir, 'buy-and-hold_spy_summary.csv')}")

    # -------------------------------
    # Strategy 2: Regime-Based Allocation (Econ)
    # -------------------------------
    regime_filename_econ = './data/predictions/1M_econ_preds.csv'
    model_result_econ, version_suffix_econ = run_regime_backtest_from_file(regime_filename_econ, returns_df, report_dir, "Regime-Based Allocation")

    # -------------------------------
    # Strategy 3: Regime-Based Allocation (Market)
    # -------------------------------
    regime_filename_mkt = './data/predictions/1M_mkt_preds.csv'
    model_result_mkt, version_suffix_mkt = run_regime_backtest_from_file(regime_filename_mkt, returns_df, report_dir, "Regime-Based Allocation")

    # -------------------------------
    # Compare Equity Curves
    # -------------------------------
    plt.figure(figsize=(12,6))
    plt.plot(result.equity_curve, label='Buy-and-Hold SPY')
    plt.plot(model_result_econ.equity_curve, label=f'Regime-Based Allocation{version_suffix_econ}')
    plt.plot(model_result_mkt.equity_curve, label=f'Regime-Based Allocation{version_suffix_mkt}')
    plt.title("Equity Curve Comparison")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(report_dir, f'equity_curve_comparison_{version_suffix_econ[1:]}_{version_suffix_mkt[1:]}.png'))

    # Save equity curve comparison as CSV for all 3 strategies
    comparison_df = pd.DataFrame({
        'Buy-and-Hold SPY': result.equity_curve,
        f'Regime-Based Allocation{version_suffix_econ}': model_result_econ.equity_curve,
        f'Regime-Based Allocation{version_suffix_mkt}': model_result_mkt.equity_curve,
    })
    comparison_df.to_csv(os.path.join(
        report_dir,
        f'equity_curve_comparison_{version_suffix_econ[1:]}_{version_suffix_mkt[1:]}.csv'
    ))

    # -------------------------------
    # Rolling Sharpe Ratio Comparison
    # -------------------------------
    window = 12  # 12-month rolling window
    rolling_sharpe = pd.DataFrame()
    rolling_sharpe['Buy-and-Hold SPY'] = result.daily_returns.rolling(window).mean() / result.daily_returns.rolling(window).std() * np.sqrt(12)
    rolling_sharpe[f'Regime-Based Allocation{version_suffix_econ}'] = model_result_econ.daily_returns.rolling(window).mean() / model_result_econ.daily_returns.rolling(window).std() * np.sqrt(12)
    rolling_sharpe[f'Regime-Based Allocation{version_suffix_mkt}'] = model_result_mkt.daily_returns.rolling(window).mean() / model_result_mkt.daily_returns.rolling(window).std() * np.sqrt(12)
    plt.figure(figsize=(12, 6))
    for col in rolling_sharpe.columns:
        plt.plot(rolling_sharpe.index, rolling_sharpe[col], label=col)
    plt.title("Rolling Sharpe Ratio (12-Month Window)")
    plt.xlabel("Date")
    plt.ylabel("Sharpe Ratio")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(report_dir, f'rolling_sharpe_comparison_{version_suffix_econ[1:]}_{version_suffix_mkt[1:]}.png'))
    rolling_sharpe.to_csv(os.path.join(report_dir, f'rolling_sharpe_comparison_{version_suffix_econ[1:]}_{version_suffix_mkt[1:]}.csv'))

    # -------------------------------
    # Compare Drawdown Curves
    # -------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(result.drawdown, label='Buy-and-Hold SPY')
    plt.plot(model_result_econ.drawdown, label=f'Regime-Based Allocation{version_suffix_econ}')
    plt.plot(model_result_mkt.drawdown, label=f'Regime-Based Allocation{version_suffix_mkt}')
    plt.title("Drawdown Curve Comparison")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(report_dir, f'drawdown_curve_comparison_{version_suffix_econ[1:]}_{version_suffix_mkt[1:]}.png'))

    # Save drawdown curves as CSV
    drawdown_df = pd.DataFrame({
        'Buy-and-Hold SPY': result.drawdown,
        f'Regime-Based Allocation{version_suffix_econ}': model_result_econ.drawdown,
        f'Regime-Based Allocation{version_suffix_mkt}': model_result_mkt.drawdown,
    })
    drawdown_df.to_csv(os.path.join(
        report_dir,
        f'drawdown_curve_comparison_{version_suffix_econ[1:]}_{version_suffix_mkt[1:]}.csv'
    ))
