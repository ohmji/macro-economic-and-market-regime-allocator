# Macro Economic and Market Regime Allocator

A machine learning framework that forecasts economic and market regimes — such as Expansion and Recession — and applies dynamic asset allocation strategies based on regime signals.

## 📈 Project Overview

This project integrates macroeconomic indicators and market-based data to:

- Predict long-term **economic regimes** (e.g., Expansion, Recession)
- Detect short-term **market stress** (e.g., Normal)
- Map regime forecasts into portfolio weights
- Backtest performance using dynamic regime-based allocation

---

## 🧠 Regime Prediction Models

The framework includes two main models:

### 🟢 Economic Regime Model (`econ`)
- Objective: Predict long-term macroeconomic conditions — *Expansion* or *Recession*
- Data: Based on macroeconomic indicators from the FRED-MD dataset
- Algorithm: XGBoost with rolling-window training
- Output: `1M_econ_preds.csv` — Monthly recession probabilities from 3 models (DT, RF, XGB)

### 🔴 Market Regime Model (`market`)
- Objective: Detect short-term market stress — *Normal* or *Crash*
- Data: Market-based indicators and lagged macroeconomic features
- Algorithm: XGBoost with rolling-window training
- Output: `1M_mkt_preds.csv` — Monthly crash probabilities from 3 models (DT, RF, XGB)

---

## ⚙️ Pipeline Structure

```bash
src/
├── allocator.py                 # Map regimes to portfolio weights
├── modeling-evaluation-econ.py # Economic regime training, evaluation, feature importance
├── modeling-evaluation-market.py # Market regime training, evaluation, feature importance
├── model_evaluation.py         # Metrics, visualizations
├── data_understanding.py       # Plot regimes timeline
data/
├── raw_data/                   # Original FRED/Moody's data
├── datasets/                   # Prepared datasets with labels
├── predictions/                # Model outputs (.csv)
├── report/                     # Evaluation plots, metrics (.png, .csv)
```

---

## 🛠️ How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run economic model training and evaluation:
```bash
python src/modeling-evaluation-econ.py
```

3. Run market model training and evaluation:
```bash
python src/modeling-evaluation-market.py
```

Outputs:
- CSV: `1M_econ_preds.csv`, `1M_mkt_preds.csv`
- Reports: `report/DT_recession_probs.png`, `report/XGB_confusion_matrix.png`, etc.

---

## 💼 Portfolio Allocation

The `allocator.py` script maps predicted regimes to asset weights:

| Regime     | SPY | TLT | GLD | BIL |
|------------|-----|-----|-----|-----|
| Expansion  | 70% | 20% | 10% |  0% |
| Recession  |  0% | 50% | 30% | 20% |

You can integrate these weights with your backtesting engine.

---

## 📊 Evaluation Metrics

- Accuracy, Precision, Recall, F1, ROC-AUC
- Confusion matrix
- Time-series plots of regime prediction probabilities
- Feature importance (XGBoost)

---

## 📁 Sample Outputs

- `./data/predictions/1M_econ_preds.csv`
- `./report/DT_confusion_matrix.png`
- `./report/XGB_recession_probs.png`

---

## 📌 Notes

- Economic labels are based on **NBER Business Cycle Dating**
- Models use **rolling-window training** for out-of-sample robustness
- Feature engineering relies on **FRED-MD** macroeconomic dataset

---

## 🔒 License

MIT License — © 2023–2025 Ali Rahimi, with modifications by contributors.

---

## 🙏 Acknowledgements

This project is inspired by and adapted from the excellent work by [ARahimiQuant](https://github.com/ARahimiQuant/forecasting-economic-and-market-regimes).  
Special thanks to their comprehensive framework for forecasting economic and market regimes, which provided valuable guidance and reference for this implementation.