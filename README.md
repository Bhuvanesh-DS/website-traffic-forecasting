# 📈 Website Traffic Forecasting

Forecasting daily website visitor traffic using time-series analysis (ARIMA) and Machine Learning (Random Forest), with a side-by-side model comparison.

---

## 📌 Problem Statement

Predicting website traffic helps businesses plan server capacity, marketing campaigns, and content scheduling. This project builds and compares two forecasting approaches — a statistical time-series model (ARIMA) and a machine learning model (Random Forest) — to predict daily unique website visitors.

---

## 📊 Dataset

- **File:** `daily-website-visitors.csv`
- **Features:** `Date`, `Unique.Visits`, `First.Time.Visits`, `Returning.Visits`
- **Target:** `Unique.Visits` (daily unique visitors)
- **Split:** 80% train / 20% test (time-series order preserved)

---

## 🛠 Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| Pandas + NumPy | Data loading & preprocessing |
| Statsmodels (ARIMA) | Time-series forecasting |
| Scikit-learn (Random Forest) | ML-based forecasting |
| Matplotlib | Visualization & EDA |
| Jupyter Notebook | Development environment |

---

## ⚙️ Methodology

1. **Data Loading** — Loaded daily visitor CSV, parsed dates
2. **Preprocessing** — Sorted by date, set Date as index, handled missing values
3. **EDA** — Plotted Unique Visits, First Time Visits, Returning Visits trends
4. **Train-Test Split** — 80/20 time-series split (no shuffle)
5. **ARIMA Model** — Fitted ARIMA(5,1,0) on training data, forecasted test period
6. **Feature Engineering** — Extracted day, month, dayofweek from date for ML
7. **Random Forest** — Trained RandomForestRegressor (200 estimators) on engineered features
8. **Model Comparison** — Compared ARIMA vs Random Forest using MAE and RMSE

---

## 📈 Model Performance

| Model | MAE | RMSE |
|---|---|---|
| ARIMA(5,1,0) | *(update with your value)* | *(update with your value)* |
| Random Forest | *(update with your value)* | *(update with your value)* |

> Run the notebook to get your exact metric values and update this table.

---

## 📊 Key Visualizations

- Daily Unique Visitors trend over time
- First Time vs Returning Visitors comparison
- ARIMA forecast vs actual traffic
- Random Forest predictions vs actual traffic

---

## 📁 Project Structure

```
website-traffic-forecasting/
├── website.ipynb                  ← Main Jupyter notebook
├── daily-website-visitors.csv     ← Dataset
├── requirements.txt               ← Python dependencies
└── README.md                      ← Project documentation
```

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/Bhuvanesh-DS/website-traffic-forecasting.git
cd website-traffic-forecasting

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open the notebook
jupyter notebook website.ipynb
```

---

## 📦 Requirements

```
pandas
numpy
matplotlib
scikit-learn
statsmodels
jupyter
```

---

## 🔮 Future Improvements

- [ ] Add Facebook Prophet for better trend + seasonality decomposition
- [ ] Add LSTM deep learning model for sequence-based forecasting
- [ ] Build a Streamlit dashboard with live traffic predictions
- [ ] Add feature importance plot for Random Forest model

---

## 👤 Author

**Bhuvanesh S**
📍 Bengaluru, India
📫 bhuvanesh9602@email.com
🔗 [LinkedIn](https://l1nk.dev/7ppsxgs) | [GitHub](https://github.com/Bhuvanesh-DS)

---

*Part of my Data Science portfolio — built during internship at First Quadrant Labs*
