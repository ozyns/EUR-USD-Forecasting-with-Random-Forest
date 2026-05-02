# EUR/USD Forecasting — Random Forest

Supervised ML pipeline to forecast the EUR/USD exchange rate using 10 years of historical data.
Built as a team of three, each owning a distinct part of the pipeline.

> Full writeup → [ozyns.github.io](https://ozyns.github.io)

---

## Results

| Metric | Value |
|--------|-------|
| R² | 0.95 |
| RMSE | 0.003105 |
| MAE | 0.002865 |
| MAPE | 0.2569% |
| Directional Accuracy | 74.76% |

---

## Pipeline

```
yfinance → EnhancedDataEngineer → OptimizedModelDeveloper → ModelEvaluator
```

| Step | Class | Description |
|------|-------|-------------|
| Data | `EnhancedDataEngineer` | Fetches EUR/USD, Gold, Oil from Yahoo Finance and engineers 36 features |
| Model | `OptimizedModelDeveloper` | Trains a Random Forest with chronological train/test split |
| Evaluation | `ModelEvaluator` | Generates prediction plots, error distribution, and feature importances |

---

## Features (36 total)

- Returns: 1d, 3d, 7d
- Volatility: rolling std over 7 and 14 days
- Moving averages + crossovers: MA(7), MA(21), MA(50)
- RSI(14) + overbought/oversold signal
- Lag features: price and return lags at 1, 2, 3, 5, 7, 14 days
- Rolling stats: 7-day high, low, range
- Temporal: day of week, month, quarter
- Exogenous: Gold and Oil returns + 7-day MAs

---

## Project Structure

```
eurusd-forecasting/
├── src/
│   ├── data_engineer.py      # EnhancedDataEngineer
│   ├── model_developer.py    # OptimizedModelDeveloper
│   └── evaluator.py          # ModelEvaluator
├── main.py                   # entry point
├── export_results.py         # generates blog assets
├── requirements.txt
└── README.md
```

---

## Getting Started

```bash
# clone
git clone https://github.com/ozyns/EUR-USD-Forecasting-with-Random-Forest
cd EUR-USD-Forecasting-with-Random-Forest

# install dependencies
pip install -r requirements.txt

# run
python main.py
```

Outputs:
- `evaluation_report.png` — 4-panel evaluation figure
- `trained_model.pkl` — saved model
- `processed_data.csv` — engineered feature set

---

## Team

| Role | Owner |
|------|-------|
| Data Engineering | [name] |
| Model Development | [name] |
| Evaluation | [name] |

---

## License

MIT
