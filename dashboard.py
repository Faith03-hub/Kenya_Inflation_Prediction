# dashboard.py - Kenya Inflation Dashboard with Model Comparison

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import warnings
warnings.filterwarnings('ignore')

from src.arima_model import run_arima
from src.prophet_model import run_prophet
from src.evaluation import evaluate_model

st.set_page_config(page_title="Kenya Inflation Dashboard", layout="wide")

# ============================================================================
# LOAD DATA
# ============================================================================

@st.cache_data
def load_data():
    url = "https://api.worldbank.org/v2/country/KE/indicator/FP.CPI.TOTL.ZG?format=json&per_page=1000"
    response = requests.get(url)
    data = response.json()
    
    df = pd.DataFrame(data[1])[['date', 'value']]
    df.columns = ['Year', 'Inflation_Rate']
    df['Year'] = pd.to_datetime(df['Year'], format='%Y')
    df['Inflation_Rate'] = pd.to_numeric(df['Inflation_Rate'], errors='coerce')
    df = df.dropna().sort_values('Year')
    
    # Filter to 1980-2024
    df = df[(df['Year'].dt.year >= 1980) & (df['Year'].dt.year <= 2024)]
    return df

df = load_data()

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title("📊 Navigation")
st.sidebar.markdown("### Kenya Inflation Dashboard")
st.sidebar.markdown(f"Data: {df['Year'].dt.year.min()} - {df['Year'].dt.year.max()}")
st.sidebar.markdown("---")
forecast_years = st.sidebar.slider("Forecast Horizon (Years)", 1, 10, 5)
st.sidebar.markdown("---")
st.sidebar.caption("Models: ARIMA & Prophet")

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

st.title("📈 Kenya Inflation Rate Dashboard")
st.markdown("**Historical Analysis (1980-2024) + Forecasts**")

# Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📅 Data Period", f"{df['Year'].dt.year.min()} - {df['Year'].dt.year.max()}")
with col2:
    st.metric("📊 Avg Inflation", f"{df['Inflation_Rate'].mean():.1f}%")
with col3:
    st.metric("🔺 Peak Inflation", f"{df['Inflation_Rate'].max():.1f}%")
with col4:
    st.metric(f"📌 Latest ({df['Year'].dt.year.max()})", f"{df['Inflation_Rate'].iloc[-1]:.1f}%")

# Historical Chart
st.subheader("📊 Historical Inflation Trend")

fig1, ax = plt.subplots(figsize=(14, 5))
ax.plot(df['Year'].dt.year, df['Inflation_Rate'], 'o-', color='#1f77b4', linewidth=2, markersize=4)
ax.axhline(y=5, color='green', linestyle='--', alpha=0.7, label='Target (5%)')
ax.set_xlabel("Year")
ax.set_ylabel("Inflation Rate (%)")
ax.set_title("Kenya Inflation Rate (1980-2024)", fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig1)

# ============================================================================
# MODEL COMPARISON SECTION (NEW)
# ============================================================================

st.subheader("🏆 Model Performance Comparison")

# Split data for evaluation (train: 1980-2019, test: 2020-2024)
train = df[df['Year'].dt.year <= 2019]
test = df[df['Year'].dt.year > 2019]

if len(test) > 0:
    with st.spinner("Evaluating models on 2020-2024 test data..."):
        # Get forecasts for test period
        arima_test, arima_order = run_arima(train, steps=len(test))
        prophet_test = run_prophet(train, steps=len(test))
        
        # Calculate errors
        arima_mae, arima_rmse = evaluate_model(test['Inflation_Rate'].values, arima_test['yhat'].values)
        prophet_mae, prophet_rmse = evaluate_model(test['Inflation_Rate'].values, prophet_test['yhat'].values)
        
        # Determine winner
        if prophet_rmse < arima_rmse:
            winner = "Prophet"
            winner_color = "#ff7f0e"
            winner_icon = "🟠"
            margin = ((arima_rmse - prophet_rmse) / arima_rmse) * 100
        else:
            winner = "ARIMA"
            winner_color = "#1f77b4"
            winner_icon = "🔵"
            margin = ((prophet_rmse - arima_rmse) / prophet_rmse) * 100
    
    # Display metrics side by side
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown("### 🔵 ARIMA Model")
        st.metric("MAE (Mean Absolute Error)", f"{arima_mae:.2f}%")
        st.metric("RMSE (Root Mean Square Error)", f"{arima_rmse:.2f}%")
    
    with col2:
        st.markdown("### 🟠 Prophet Model")
        st.metric("MAE (Mean Absolute Error)", f"{prophet_mae:.2f}%")
        st.metric("RMSE (Root Mean Square Error)", f"{prophet_rmse:.2f}%")
    
    with col3:
        st.markdown("### 🏆 Winner")
        st.markdown(f"## {winner_icon} {winner}")
        st.caption(f"{margin:.1f}% better RMSE")
    
    # Actual vs Predicted chart
    fig_compare, ax = plt.subplots(figsize=(12, 5))
    
    years = test['Year'].dt.year.values
    ax.plot(years, test['Inflation_Rate'].values, 'o-', label='Actual', color='black', linewidth=2, markersize=8)
    ax.plot(years, arima_test['yhat'].values, 's--', label=f'ARIMA{arima_order}', color='#1f77b4', linewidth=2, markersize=6)
    ax.plot(years, prophet_test['yhat'].values, 'D--', label='Prophet', color='#ff7f0e', linewidth=2, markersize=6)
    
    ax.set_xlabel("Year")
    ax.set_ylabel("Inflation Rate (%)")
    ax.set_title("Model Performance: Actual vs Predicted (2020-2024)", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig_compare)
    
    # Summary explanation
    st.info(f"""
    **📊 Summary:** On the 2020-2024 test data, **{winner}** performed better with 
    **{margin:.1f}% lower RMSE**. This means {winner} predictions were closer to actual inflation values.
    """)

else:
    st.warning("Insufficient test data for model comparison")

st.markdown("---")

# ============================================================================
# FORECAST
# ============================================================================

st.subheader("🔮 Forecast (2025-2029)")

with st.spinner("Generating forecasts..."):
    forecast_arima, arima_order = run_arima(df, forecast_years)
    forecast_prophet = run_prophet(df, forecast_years)

# Display results
col1, col2 = st.columns(2)

with col1:
    st.write(f"**ARIMA{arima_order} Forecast**")
    arima_df = pd.DataFrame({
        "Year": forecast_arima["ds"].dt.year,
        "Forecast (%)": forecast_arima["yhat"].round(2),
        "Lower 95%": forecast_arima["yhat_lower"].round(2),
        "Upper 95%": forecast_arima["yhat_upper"].round(2)
    })
    st.dataframe(arima_df, hide_index=True, use_container_width=True)

with col2:
    st.write("**Prophet Forecast**")
    prophet_df = pd.DataFrame({
        "Year": forecast_prophet["ds"].dt.year,
        "Forecast (%)": forecast_prophet["yhat"].round(2),
        "Lower 95%": forecast_prophet["yhat_lower"].round(2),
        "Upper 95%": forecast_prophet["yhat_upper"].round(2)
    })
    st.dataframe(prophet_df, hide_index=True, use_container_width=True)

# Forecast chart
fig2, ax = plt.subplots(figsize=(14, 5))

# Historical
ax.plot(df['Year'].dt.year, df['Inflation_Rate'], 'o-', label="Historical", color='gray', linewidth=2, markersize=4)

# ARIMA
ax.plot(forecast_arima["ds"].dt.year, forecast_arima["yhat"], 's--', label=f"ARIMA{arima_order}", color='#1f77b4', linewidth=2, markersize=6)
ax.fill_between(forecast_arima["ds"].dt.year, forecast_arima["yhat_lower"], forecast_arima["yhat_upper"], 
                color='#1f77b4', alpha=0.2)

# Prophet
ax.plot(forecast_prophet["ds"].dt.year, forecast_prophet["yhat"], 'D--', label="Prophet", color='#ff7f0e', linewidth=2, markersize=6)
ax.fill_between(forecast_prophet["ds"].dt.year, forecast_prophet["yhat_lower"], forecast_prophet["yhat_upper"], 
                color='#ff7f0e', alpha=0.2)

ax.axhline(y=5, color='green', linestyle=':', linewidth=2, label='Target (5%)')
ax.axvline(x=2024.5, color='red', linestyle='--', alpha=0.5, label='Forecast Start')
ax.set_xlabel("Year")
ax.set_ylabel("Inflation Rate (%)")
ax.set_title("Kenya Inflation: Historical + Forecast (2025-2029)", fontsize=14, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
st.pyplot(fig2)

# ============================================================================
# DOWNLOAD
# ============================================================================

st.subheader("📥 Download Forecasts")

col1, col2 = st.columns(2)
with col1:
    st.download_button("Download ARIMA Forecast", arima_df.to_csv(index=False), "arima_forecast.csv")
with col2:
    st.download_button("Download Prophet Forecast", prophet_df.to_csv(index=False), "prophet_forecast.csv")

# Raw data expander
with st.expander("📄 View Raw Historical Data"):
    st.dataframe(df)

st.caption("Data: World Bank API | Models: ARIMA & Prophet | Evaluation: 2020-2024 test period")
