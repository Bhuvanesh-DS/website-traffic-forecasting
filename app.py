import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Website Traffic Forecasting",
    page_icon="📈",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────
st.title("📈 Website Traffic Forecasting")
st.markdown("**Forecasting daily website visitors using ARIMA & Random Forest**")
st.markdown("*Built by [Bhuvanesh S](https://github.com/Bhuvanesh-DS) — Data Science Portfolio Project*")
st.divider()

# ── File Upload ───────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
train_split = st.sidebar.slider("Train/Test Split (%)", 60, 90, 80)
arima_order_p = st.sidebar.slider("ARIMA p value", 1, 10, 5)
n_estimators = st.sidebar.slider("Random Forest Trees", 50, 300, 200, step=50)

st.sidebar.divider()
st.sidebar.markdown("**About this app**")
st.sidebar.markdown("This app compares ARIMA (statistical) vs Random Forest (ML) for time-series forecasting of website traffic.")

# ── Load Data ─────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)
    # Clean comma-formatted numbers
    for col in ['Unique.Visits', 'First.Time.Visits', 'Returning.Visits']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    return df

if uploaded_file is not None:
    df = load_data(uploaded_file)

    # ── Dataset Overview ──────────────────────────────────────
    st.subheader("📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Date Range", f"{df.index.min().year} – {df.index.max().year}")
    col3.metric("Avg Daily Visitors", f"{df['Unique.Visits'].mean():,.0f}")
    col4.metric("Peak Visitors", f"{df['Unique.Visits'].max():,.0f}")

    with st.expander("Preview raw data"):
        st.dataframe(df.head(20), use_container_width=True)

    st.divider()

    # ── EDA Plots ─────────────────────────────────────────────
    st.subheader("🔍 Exploratory Data Analysis")
    tab1, tab2, tab3 = st.tabs(["Unique Visits", "First Time Visits", "Returning Visits"])

    with tab1:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df.index, df['Unique.Visits'], color='#1f77b4', linewidth=1)
        ax.set_title("Daily Unique Visitors Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Unique Visits")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df.index, df['First.Time.Visits'], color='#2ca02c', linewidth=1)
        ax.set_title("First Time Visitors Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("First Time Visits")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with tab3:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df.index, df['Returning.Visits'], color='#ff7f0e', linewidth=1)
        ax.set_title("Returning Visitors Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Returning Visits")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    st.divider()

    # ── Train/Test Split ──────────────────────────────────────
    train_size = int(len(df) * train_split / 100)
    train = df.iloc[:train_size].copy()
    test = df.iloc[train_size:].copy()

    st.subheader("🤖 Model Training & Forecasting")
    col1, col2 = st.columns(2)
    col1.metric("Training samples", f"{len(train):,}")
    col2.metric("Testing samples", f"{len(test):,}")

    # ── ARIMA ─────────────────────────────────────────────────
    with st.spinner("Training ARIMA model..."):
        train_arima = train.copy().reset_index(drop=True)
        test_arima = test.copy().reset_index(drop=True)
        y_true = pd.to_numeric(test_arima['Unique.Visits'].astype(str).str.replace(',', ''), errors='coerce')
        train_vals = pd.to_numeric(train_arima['Unique.Visits'].astype(str).str.replace(',', ''), errors='coerce').dropna()
        arima_model = ARIMA(train_vals, order=(arima_order_p, 1, 0))
        arima_fit = arima_model.fit()
        arima_forecast = arima_fit.forecast(steps=len(test))
        arima_mae = mean_absolute_error(y_true, arima_forecast)
        arima_rmse = np.sqrt(mean_squared_error(y_true, arima_forecast))

    # ── Random Forest ─────────────────────────────────────────
    with st.spinner("Training Random Forest model..."):
        df_ml = df.copy()
        df_ml['Unique.Visits'] = pd.to_numeric(df_ml['Unique.Visits'].astype(str).str.replace(',', ''), errors='coerce')
        df_ml['day'] = df_ml.index.day
        df_ml['month'] = df_ml.index.month
        df_ml['dayofweek'] = df_ml.index.dayofweek
        X = df_ml[['day', 'month', 'dayofweek']]
        y = df_ml['Unique.Visits']
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_preds = rf_model.predict(X_test)
        rf_mae = mean_absolute_error(y_test, rf_preds)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))

    st.success("✅ Both models trained successfully!")
    st.divider()

    # ── Results ───────────────────────────────────────────────
    st.subheader("📊 Model Comparison")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📉 ARIMA Model")
        m1, m2 = st.columns(2)
        m1.metric("MAE", f"{arima_mae:,.2f}")
        m2.metric("RMSE", f"{arima_rmse:,.2f}")

    with col2:
        st.markdown("### 🌲 Random Forest")
        m1, m2 = st.columns(2)
        m1.metric("MAE", f"{rf_mae:,.2f}")
        m2.metric("RMSE", f"{rf_rmse:,.2f}")

    # Winner
    winner = "ARIMA" if arima_mae < rf_mae else "Random Forest"
    st.info(f"🏆 **{winner}** performs better on this dataset (lower MAE)")

    st.divider()

    # ── Forecast Plots ────────────────────────────────────────
    st.subheader("📈 Forecast vs Actual")
    tab1, tab2 = st.tabs(["ARIMA Forecast", "Random Forest Forecast"])

    with tab1:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(range(len(train_vals)), train_vals, label="Training Data", color='#1f77b4', linewidth=1)
        ax.plot(range(len(train_vals), len(train_vals) + len(y_true)), y_true, label="Actual", color='#2ca02c', linewidth=1.5)
        ax.plot(range(len(train_vals), len(train_vals) + len(arima_forecast)), arima_forecast, label="ARIMA Forecast", color='#d62728', linewidth=1.5, linestyle='--')
        ax.set_title(f"ARIMA({arima_order_p},1,0) Forecast vs Actual")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Unique Visits")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(range(len(y_test)), y_test.values, label="Actual", color='#2ca02c', linewidth=1.5)
        ax.plot(range(len(rf_preds)), rf_preds, label="RF Forecast", color='#9467bd', linewidth=1.5, linestyle='--')
        ax.set_title("Random Forest Forecast vs Actual")
        ax.set_xlabel("Test Samples")
        ax.set_ylabel("Unique Visits")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    st.divider()
    st.markdown("*Built by Bhuvanesh S | [GitHub](https://github.com/Bhuvanesh-DS) | [LinkedIn](https://l1nk.dev/7ppsxgs)*")

else:
    # ── Landing screen ────────────────────────────────────────
    st.info("👈 Upload your `daily-website-visitors.csv` file from the sidebar to get started!")
    st.markdown("""
    ### What this app does:
    - 📊 **Visualizes** daily website traffic trends (Unique, First-time, Returning visitors)
    - 🤖 **Trains two models**: ARIMA (statistical) and Random Forest (ML)
    - 📈 **Compares** both models side by side with MAE and RMSE metrics
    - 🏆 **Declares a winner** based on prediction accuracy

    ### How to use:
    1. Upload the `daily-website-visitors.csv` file in the sidebar
    2. Adjust train/test split and model parameters
    3. View EDA charts, forecasts, and model comparison

    ---
    *Built by Bhuvanesh S as part of Data Science portfolio*
    """)
