# Macro Regime Allocator

This project builds a machine learning-based economic regime detection model, and uses it to dynamically allocate assets in a portfolio based on macroeconomic conditions.

## 💡 Project Objective

- Classify macroeconomic regimes (Expansion, Recession, Slowdown, Recovery) using ML models (e.g., XGBoost)
- Allocate portfolios dynamically based on predicted regimes
- Outperform traditional 60/40 strategies with lower drawdowns

## 📈 Economic Indicators Used

- Yield Curve (10Y-2Y)
- ISM PMI
- Unemployment Rate
- Core CPI (YoY)
- Retail Sales
- Fed Balance Sheet

## 🔧 Methodology

1. **Data Pipeline** — Pull data from FRED and other APIs
2. **Feature Engineering** — Create lagged, rolling, and normalized macro features
3. **Regime Labeling** — Use rules or clustering to define regimes
4. **Modeling** — Train classifiers (Decision Tree, Random Forest, XGBoost) to predict market and economic regimes
5. **Allocation Strategy** — Map regimes to portfolio weights and backtest
6. **Dashboard (optional)** — Streamlit app to visualize regime and allocation

### 🧠 Economic vs Market Regime Models

- **Economic Regime Model (`econ`)**:  
  Trained on macroeconomic indicators from the FRED-MD dataset (e.g., employment, inflation, consumption).  
  Uses rolling-window ML predictions (Decision Tree, Random Forest, XGBoost) to classify periods as Expansion or Recession.

- **Market Regime Model (`market`)**:  
  Trained on market regime dataset constructed from market-sensitive indicators and FRED-MD features.  
  Predicts binary market stress states (Normal vs. Crash) using time-series cross-validation and rolling XGBoost forecasts.

## 📊 Output

- Regime classification accuracy (Econ & Market)
- Portfolio Sharpe ratio and drawdowns
- Regime dashboard for real-time economic condition tracking

## 🔍 Example

| Regime     | Allocation                             |
|------------|----------------------------------------|
| Expansion  | 70% Equity, 20% Bonds, 10% Gold        |
| Recession  | 30% Gold, 50% Bonds, 20% Cash          |

*Regime classification guides portfolio rebalancing based on both macroeconomic and market signals.*

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🧪 Backtest & Evaluation

- Buy-and-hold SPY vs. Regime-based strategies
- Equity curve comparisons
- Sharpe ratio, CAGR, drawdown analysis
- Model evaluation includes confusion matrix, feature importance, and rolling predictions

## 🙏 Acknowledgements

This project is inspired by and adapted from the excellent work by [ARahimiQuant](https://github.com/ARahimiQuant/forecasting-economic-and-market-regimes).  
Special thanks to their comprehensive framework for forecasting economic and market regimes, which provided valuable guidance and reference for this implementation.