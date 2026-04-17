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


# LOAD DATA


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
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Kenya Inflation Dashboard", layout="wide")

st.title("📈 Kenya Inflation Rate Dashboard")
st.markdown("Historical Analysis & Forecast (1980-2024 + Forecasts)")

# Load existing data 
@st.cache_data
def load_data():
    # Using the same CSV I used in your notebook
    df = pd.read_csv('C:/Users/Administrator/Downloads/MY NEW PRO/kenya-gdp-forecasting/data/processed/kenya_inflation_cleaned.csv')
    if 'Year' in df.columns:
        # Checking if Year is already datetime
        if not pd.api.types.is_datetime64_any_dtype(df['Year']):
            # Handle different formats
            try:
                df['Year'] = pd.to_datetime(df['Year'])
            except:
                # If Year is just years as integers
                df['Year'] = pd.to_datetime(df['Year'], format='%Y')
 
    return df

df = load_data()

# Sidebar
st.sidebar.header("⚙️ Settings")
forecast_years = st.sidebar.slider("Forecast Horizon (Years)", 1, 10, 5)

# Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📅 Data Period", f"1980 - {df['Year'].dt.year.max()}")
with col2:
    st.metric("📊 Avg Inflation", f"{df['Inflation_Rate'].mean():.1f}%")
with col3:
    st.metric("🔺 Peak Inflation", f"{df['Inflation_Rate'].max():.1f}%")
with col4:
    current = df[df['Year'].dt.year == 2024]['Inflation_Rate'].values
    st.metric("📌 2024 Inflation", f"{current[0]:.1f}%" if len(current) > 0 else "N/A")

# Historical Chart
st.subheader("📊 Historical Inflation Trend")
fig1, ax = plt.subplots(figsize=(12, 5))
ax.plot(df['Year'].dt.year, df['Inflation_Rate'], 
        marker='o', color='#1f77b4', linewidth=2)
ax.axhline(y=5, color='green', linestyle='--', alpha=0.7, label='Target (5%)')
ax.set_title("Kenya Inflation Rate (1980-2024)")
ax.set_xlabel("Year")
ax.set_ylabel("Inflation Rate (%)")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig1)

# ARIMA Forecast
# ARIMA Forecast with Auto-Tuning (Level playing field)
# ARIMA Forecast with Train/Test Split (FAIR COMPARISON)
# ARIMA Forecast with Train/Test Split (FAIR COMPARISON)
@st.cache_resource
def run_arima(df, steps):
    from pmdarima import auto_arima
    
    # Create a copy similar to Prophet approach
    arima_df = df[['Year', 'Inflation_Rate']].copy()
    
    # SAME as Prophet: Train on 1980-2019 ONLY
    train_mask = arima_df['Year'].dt.year <= 2019
    train_data = arima_df[train_mask]['Inflation_Rate'].values
    
    # Auto-tune to find best order
    auto_model = auto_arima(
        train_data,
        start_p=0, start_q=0,
        max_p=5, max_q=5,
        seasonal=False,
        stepwise=True,
        trace=False,
        information_criterion='aic'
    )
    
    best_order = auto_model.order
    
    # Fit model on training data only
    model = ARIMA(train_data, order=best_order)
    results = model.fit()
    
    # FIX: Forecast starting from 2025 (not 2020)
    last_actual_year = df['Year'].dt.year.max()  # This gets 2024
    forecast_values = results.forecast(steps=steps)
    forecast_result = results.get_forecast(steps=steps)
    conf_int = forecast_result.conf_int()
    
    # Return DataFrame with correct years (2025-2029)
    forecast_df = pd.DataFrame({
        'ds': pd.date_range(start=f'{last_actual_year + 1}-01-01', periods=steps, freq='YS'),
        'yhat': forecast_values,
        'yhat_lower': conf_int[:, 0],
        'yhat_upper': conf_int[:, 1]
    })
    
    return forecast_df, best_order
# Prophet Forecast
@st.cache_resource
def run_prophet(df, steps):
    # Making a copy to avoid modifying original
    prophet_df = df[['Year', 'Inflation_Rate']].copy()
    prophet_df.columns = ['ds', 'y']
    
    # Set the training data to 1980-2019 
    train = prophet_df[prophet_df['ds'].dt.year <= 2019].copy()
    
    # Use EXACT SAME parameters as your tuned Jupyter model
    model = Prophet(
        changepoint_prior_scale=0.08,      
        seasonality_prior_scale=0.5,      
        yearly_seasonality=False,          
        seasonality_mode='additive',       
        interval_width=0.95                
    )
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Fit the model on training data
    model.fit(train)
    
    #  Create future dates starting from 2025 (the year after the last actual data point)
    last_actual_year = df['Year'].dt.year.max()  # This gets 2024
    future_dates = pd.date_range(start=f'{last_actual_year + 1}-01-01', periods=steps, freq='YS')
    future = pd.DataFrame({'ds': future_dates})
    
    # Predict
    forecast = model.predict(future)
    
    # Return only forecast years
    forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    return forecast


# Generate forecasts
# Generate forecasts
# Generate forecasts (both take the full DataFrame now)
with st.spinner("Generating forecasts..."):
    forecast_arima, arima_order = run_arima(df, forecast_years)
    forecast_prophet = run_prophet(df, forecast_years)

forecast_years_list = [2025 + i for i in range(forecast_years)]

# Display forecasts
st.subheader("🔮 Inflation Forecast")
col5, col6 = st.columns(2)

with col5:
    st.write(f"**ARIMA{arima_order} Forecast** (Auto-tuned)")
    arima_df = pd.DataFrame({
        'Year': forecast_arima['ds'].dt.year,
        'Forecast (%)': forecast_arima['yhat'].round(2)
    })
    st.dataframe(arima_df, hide_index=True)

with col6:
    st.write("**Prophet Forecast** (Tuned)")
    prophet_df = pd.DataFrame({
        'Year': forecast_prophet['ds'].dt.year,
        'Forecast (%)': forecast_prophet['yhat'].round(2)
    })
    st.dataframe(prophet_df, hide_index=True)

# Comparison Chart
# Comparison Chart
st.subheader("📈 Historical vs Forecast Comparison")
fig2, ax = plt.subplots(figsize=(12, 5))

# Historical
ax.plot(df['Year'].dt.year, df['Inflation_Rate'], 
        'o-', label='Historical', color='gray', linewidth=2)

# ARIMA
ax.plot(forecast_arima['ds'].dt.year, forecast_arima['yhat'], 
        '--s', label=f'ARIMA{arima_order} Forecast', color='#1f77b4', linewidth=2)

# Prophet
ax.plot(forecast_prophet['ds'].dt.year, forecast_prophet['yhat'], 
        '--D', label='Prophet Forecast', color='#ff7f0e', linewidth=2)

# Confidence intervals - FIXED: use forecast DataFrames
ax.fill_between(forecast_arima['ds'].dt.year, 
                forecast_arima['yhat_lower'], 
                forecast_arima['yhat_upper'], 
                color='#1f77b4', alpha=0.2)

ax.fill_between(forecast_prophet['ds'].dt.year, 
                forecast_prophet['yhat_lower'], 
                forecast_prophet['yhat_upper'], 
                color='#ff7f0e', alpha=0.2)

# Reference lines
ax.axvline(x=2024.5, color='gray', linestyle='--', alpha=0.5)
ax.axhline(y=5, color='green', linestyle=':', alpha=0.7, label='Target (5%)')

ax.set_title(f"Kenya Inflation: Historical + {forecast_years}-Year Forecast")
ax.set_xlabel("Year")
ax.set_ylabel("Inflation Rate (%)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(1980, 2024 + forecast_years + 1)

st.pyplot(fig2)
# Download buttons
st.subheader("📥 Export Forecasts")
col7, col8 = st.columns(2)

with col7:
    csv_arima = arima_df.to_csv(index=False)
    st.download_button("Download ARIMA Forecast", csv_arima, "arima_forecast.csv")

with col8:
    csv_prophet = prophet_df.to_csv(index=False)
    st.download_button("Download Prophet Forecast", csv_prophet, "prophet_forecast.csv")

# Raw Data Preview Section
with st.expander("🔍 View Raw Historical Data"):
    st.write("This shows the data currently loaded into the dashboard.")
    st.dataframe(df, use_container_width=True)
    
    # Add a quick summary of the data types to verify the 'Year' conversion
    st.write("**Data Column Info:**")
    st.write(df.dtypes.astype(str))
st.markdown("---")
# ============================================
# 2025: Actual vs ARIMA vs Prophet
# ============================================

st.subheader("🎯 2025 Inflation: Actual vs Forecasts")

# Actual 2025 inflation (from KNBS)
actual_2025 = 4.45

# Get forecasts from your dataframes
arima_2025 = arima_df[arima_df['Year'] == 2025]['Forecast (%)'].values[0]
prophet_2025 = prophet_df[prophet_df['Year'] == 2025]['Forecast (%)'].values[0]

# Calculate errors
arima_error = arima_2025 - actual_2025
prophet_error = prophet_2025 - actual_2025

# Display metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("📊 Actual 2025", f"{actual_2025}%")

with col2:
    st.metric("📈 ARIMA Forecast", f"{arima_2025:.2f}%", 
              delta=f"{arima_error:+.2f}%", delta_color="inverse")

with col3:
    st.metric("🔮 Prophet Forecast", f"{prophet_2025:.2f}%", 
              delta=f"{prophet_error:+.2f}%", delta_color="inverse")

# Comparison table
comparison_df = pd.DataFrame({
    "Model": ["ARIMA", "Prophet"],
    "Forecast (%)": [f"{arima_2025:.2f}", f"{prophet_2025:.2f}"],
    "Actual (%)": [f"{actual_2025}", f"{actual_2025}"],
    "Error (%)": [f"{arima_error:+.2f}", f"{prophet_error:+.2f}"],
    "|Error| (%)": [f"{abs(arima_error):.2f}", f"{abs(prophet_error):.2f}"]
})
st.dataframe(comparison_df, hide_index=True, use_container_width=True)

# Professional Bar Chart
fig, ax = plt.subplots(figsize=(6, 5))

models = ["ARIMA", "Prophet", "Actual"]
values = [arima_2025, prophet_2025, actual_2025]
colors = ['#E63946', '#2A9D8F', '#264653']  # Professional color palette
bars = ax.bar(models, values, color=colors, edgecolor='white', linewidth=2, alpha=0.85)

# Add value labels on top of bars
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15, 
            f'{val:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Customize chart
ax.set_ylabel('Inflation Rate (%)', fontsize=12, fontweight='semibold')
ax.set_title('2025 Inflation: Actual vs Forecasts', fontsize=14, fontweight='bold', pad=15)
ax.set_ylim(0, max(values) + 1.5)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add a horizontal line at target (5%)
ax.axhline(y=5, color='gray', linestyle=':', alpha=0.7, label='CBK Target (5%)')
ax.legend(loc='upper right')

plt.tight_layout()
st.pyplot(fig)

# Winner
winner = "Prophet" if abs(prophet_error) < abs(arima_error) else "ARIMA"
winner_error = min(abs(prophet_error), abs(arima_error))

if winner_error <= 1:
    st.success(f"✅ **Winner: {winner}** (Error: {winner_error:.2f}%) - Excellent!")
elif winner_error <= 2:
    st.warning(f"⚠️ **Winner: {winner}** (Error: {winner_error:.2f}%) - Good.")
else:
    st.error(f"❌ **Winner: {winner}** (Error: {winner_error:.2f}%) - Needs improvement.")
# Footer
st.markdown("---")
st.caption("📊 **Data:** World Bank (1980-2024) | **Forecast:** Prophet & ARIMA")
st.caption("📌 2025-2029 are model predictions. World Bank official data has 2-year lag. Actual 2025: 4.45% (KNBS).")
st.caption("Source: WORLD BANK | Models: ARIMA(0,0,1) & Tuned Prophet") 


