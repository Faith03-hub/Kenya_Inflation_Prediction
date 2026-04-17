# pages/Upload.py - Complete Working Version with Model Evaluation

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arima_model import run_arima
from src.prophet_model import run_prophet

st.set_page_config(page_title="Upload & Forecast", layout="wide")
st.title("📤 Upload Data & Forecast")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    
    # Read file with better date handling
    if uploaded_file.name.endswith('.csv'):
        # Try multiple approaches to read CSV with dates
        try:
            # First attempt: auto-detect dates
            df = pd.read_csv(uploaded_file, parse_dates=True)
        except:
            # Second attempt: read as is
            df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.subheader("Raw Data Preview")
    st.dataframe(df.head())
    
    # Show column names to help debug
    st.write("📊 **Columns found:**", list(df.columns))
    
    # COUNTRY FILTER SECTION - ONLY SHOW IF COUNTRY COLUMN EXISTS
    # Check if country column exists (case insensitive)
    country_col = None
    for col in df.columns:
        if col.lower() == 'country':
            country_col = col
            break
    
    # Conditional display of country filter
    if country_col:
        st.subheader("🌍 Country Selection")
        
        # Get unique countries
        unique_countries = df[country_col].unique()
        
        # Sort countries alphabetically
        unique_countries = sorted(unique_countries)
        
        # Create country selection dropdown
        selected_country = st.selectbox(
            "Select Country to Analyze",
            options=unique_countries,
            help="Choose a country to filter the data"
        )
        
        # Filter data for selected country
        df_filtered = df[df[country_col] == selected_country].copy()
        
        st.success(f"📌 Showing data for **{selected_country}** ({len(df_filtered)} records)")
        
        # Show country statistics if inflation column exists
        infl_col_country = None
        for col in df_filtered.columns:
            if 'inflation' in col.lower() or 'rate' in col.lower():
                infl_col_country = col
                break
        
        if infl_col_country:
            avg_inflation = df_filtered[infl_col_country].mean()
            st.metric(f"Average Inflation for {selected_country}", f"{avg_inflation:.2f}%")
        
    else:
        # If no country column, use all data without country filter UI
        df_filtered = df.copy()
        selected_country = "All Data"
        # Don't show any country selection UI
        st.info("📌 No country column detected. Analyzing all data.")
    
    # Find year and inflation columns
    year_col = None
    infl_col = None
    
    for col in df_filtered.columns:
        col_lower = col.lower()
        if 'year' in col_lower or 'date' in col_lower:
            if year_col is None:  # Take the first matching column
                year_col = col
        if 'inflation' in col_lower or 'rate' in col_lower:
            if infl_col is None:  # Take the first matching column
                infl_col = col
    
    # Manual column selection if automatic detection fails
    if year_col is None:
        year_col = st.selectbox("Select Year column", df_filtered.columns)
    if infl_col is None:
        infl_col = st.selectbox("Select Inflation column", df_filtered.columns)
    
    # Process data with improved date handling
    try:
        processed = pd.DataFrame()
        
        # Handle year/date column with multiple formats
        year_data = df_filtered[year_col]
        
        # Try different date formats
        if pd.api.types.is_numeric_dtype(year_data):
            # If numeric (e.g., 2024), convert directly
            processed['Year'] = pd.to_numeric(year_data, errors='coerce')
        else:
            # If string/object, try to extract year
            try:
                # Try parsing as datetime first
                processed['Year'] = pd.to_datetime(year_data, errors='coerce').dt.year
            except:
                # If that fails, try extracting year from string
                processed['Year'] = year_data.astype(str).str.extract(r'(\d{4})').astype(float)
        
        # Process inflation column
        processed['Inflation_Rate'] = pd.to_numeric(df_filtered[infl_col], errors='coerce')
        
        # Remove NaN values
        processed = processed.dropna()
        
        # Show data info before filtering
        if len(processed) > 0:
            data_min_year = int(processed['Year'].min())
            data_max_year = int(processed['Year'].max())
            st.info(f"📅 Data available from {data_min_year} to {data_max_year}")
        
        # Convert Year to datetime for consistency
        processed['Year'] = pd.to_datetime(processed['Year'], format='%Y', errors='coerce')
        processed = processed.dropna()
        processed = processed.sort_values('Year')
        
        # Dynamic year range filter based on actual data
        if len(processed) > 0:
            actual_min_year = processed['Year'].dt.year.min()
            actual_max_year = processed['Year'].dt.year.max()
        else:
            actual_min_year = 1980
            actual_max_year = 2024
        
        # Year range filter in sidebar
        st.sidebar.subheader("📅 Year Range Filter")
        year_min = st.sidebar.number_input(
            "Minimum Year", 
            min_value=int(actual_min_year), 
            max_value=int(actual_max_year), 
            value=int(actual_min_year)
        )
        year_max = st.sidebar.number_input(
            "Maximum Year", 
            min_value=int(actual_min_year), 
            max_value=int(actual_max_year), 
            value=int(actual_max_year)
        )
        
        # Apply year filter
        processed = processed[(processed['Year'].dt.year >= year_min) & (processed['Year'].dt.year <= year_max)]
        
        if len(processed) == 0:
            st.error(f"⚠️ No data found for {selected_country} in the selected year range ({year_min}-{year_max}).")
            st.info(f"Available data years: {actual_min_year}-{actual_max_year}")
            st.stop()
        
        st.success(f"✅ Loaded {len(processed)} years of data for {selected_country} ({processed['Year'].dt.year.min()}-{processed['Year'].dt.year.max()})")
        
        # Show cleaned data
        st.subheader(f"Cleaned Data - {selected_country}")
        display_df = pd.DataFrame({
            'Year': processed['Year'].dt.year,
            'Inflation Rate': processed['Inflation_Rate'].round(2)
        })
        st.dataframe(display_df)
        
        # Basic statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Inflation", f"{processed['Inflation_Rate'].mean():.2f}%")
        with col2:
            st.metric("Max Inflation", f"{processed['Inflation_Rate'].max():.2f}%")
        with col3:
            st.metric("Min Inflation", f"{processed['Inflation_Rate'].min():.2f}%")
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(processed['Year'].dt.year, processed['Inflation_Rate'], 'o-', linewidth=2, markersize=6)
        ax.set_xlabel("Year")
        ax.set_ylabel("Inflation Rate (%)")
        ax.set_title(f"Inflation Trend - {selected_country}")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Forecast Settings
        st.subheader("📈 Forecast Settings")
        
        # Add option for train/test split for model evaluation
        evaluation_split = st.slider(
            "Training data percentage (for model evaluation)", 
            min_value=50, 
            max_value=90, 
            value=80,
            help="Percentage of data to use for training, remaining for testing model accuracy"
        )
        
        forecast_years = st.slider("Years to forecast", 1, 10, 5)
        
        if st.button("Run Forecast", type="primary"):
            
            # Split data for evaluation
            train_size = int(len(processed) * evaluation_split / 100)
            train_data = processed.iloc[:train_size]
            test_data = processed.iloc[train_size:]
            
            st.info(f"📊 Model Evaluation: Using {len(train_data)} years for training, {len(test_data)} years for testing")
            
            col1, col2 = st.columns(2)
            
            # Store forecasts for evaluation
            arima_forecast = None
            prophet_forecast = None
            arima_test_predictions = None
            prophet_test_predictions = None
            
            # ARIMA
            with st.spinner("Running ARIMA model..."):
                try:
                    # Train ARIMA on training data
                    arima_forecast, order = run_arima(train_data, steps=forecast_years)
                    
                    # Generate predictions for test period for evaluation
                    if len(test_data) > 0:
                        arima_test_forecast, _ = run_arima(train_data, steps=len(test_data))
                        arima_test_predictions = arima_test_forecast['yhat'].values
                    
                    with col1:
                        st.success(f"**ARIMA{order} Forecast for {selected_country}**")
                        arima_df = pd.DataFrame({
                            'Year': arima_forecast['ds'].dt.year,
                            'Forecast': arima_forecast['yhat'].round(2),
                            'Lower': arima_forecast['yhat_lower'].round(2),
                            'Upper': arima_forecast['yhat_upper'].round(2)
                        })
                        st.dataframe(arima_df, hide_index=True)
                except Exception as e:
                    st.error(f"ARIMA error: {e}")
                    arima_forecast = None
            
            # Prophet
            with st.spinner("Running Prophet model..."):
                try:
                    # Train Prophet on training data
                    prophet_forecast = run_prophet(train_data, steps=forecast_years)
                    
                    # Generate predictions for test period for evaluation
                    if len(test_data) > 0:
                        prophet_test_forecast = run_prophet(train_data, steps=len(test_data))
                        prophet_test_predictions = prophet_test_forecast['yhat'].values
                    
                    with col2:
                        st.success(f"**Prophet Forecast for {selected_country}**")
                        prophet_df = pd.DataFrame({
                            'Year': prophet_forecast['ds'].dt.year,
                            'Forecast': prophet_forecast['yhat'].round(2),
                            'Lower': prophet_forecast['yhat_lower'].round(2),
                            'Upper': prophet_forecast['yhat_upper'].round(2)
                        })
                        st.dataframe(prophet_df, hide_index=True)
                except Exception as e:
                    st.error(f"Prophet error: {e}")
                    prophet_forecast = None
            
            # MODEL EVALUATION SECTION
            if len(test_data) > 0 and (arima_test_predictions is not None or prophet_test_predictions is not None):
                st.subheader("📊 Model Evaluation Metrics")
                st.markdown("---")
                
                # Create evaluation metrics
                eval_col1, eval_col2 = st.columns(2)
                
                # Prepare actual test values
                actual_test_values = test_data['Inflation_Rate'].values
                test_years = test_data['Year'].dt.year.values
                
                # ARIMA Evaluation
                if arima_test_predictions is not None and len(arima_test_predictions) == len(actual_test_values):
                    with eval_col1:
                        st.markdown("### 🔵 ARIMA Model Performance")
                        
                        # Calculate metrics
                        mae_arima = mean_absolute_error(actual_test_values, arima_test_predictions)
                        rmse_arima = np.sqrt(mean_squared_error(actual_test_values, arima_test_predictions))
                        mape_arima = mean_absolute_percentage_error(actual_test_values, arima_test_predictions) * 100
                        
                        # Display metrics
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        with metric_col1:
                            st.metric("MAE", f"{mae_arima:.2f}%", help="Mean Absolute Error")
                        with metric_col2:
                            st.metric("RMSE", f"{rmse_arima:.2f}%", help="Root Mean Square Error")
                        with metric_col3:
                            st.metric("MAPE", f"{mape_arima:.1f}%", help="Mean Absolute Percentage Error")
                        
                        # Create comparison dataframe
                        arima_comparison = pd.DataFrame({
                            'Year': test_years,
                            'Actual': actual_test_values.round(2),
                            'Predicted': arima_test_predictions.round(2),
                            'Error': (actual_test_values - arima_test_predictions).round(2),
                            'Absolute Error': np.abs(actual_test_values - arima_test_predictions).round(2)
                        })
                        st.dataframe(arima_comparison, hide_index=True)
                
                # Prophet Evaluation
                if prophet_test_predictions is not None and len(prophet_test_predictions) == len(actual_test_values):
                    with eval_col2:
                        st.markdown("### 🟠 Prophet Model Performance")
                        
                        # Calculate metrics
                        mae_prophet = mean_absolute_error(actual_test_values, prophet_test_predictions)
                        rmse_prophet = np.sqrt(mean_squared_error(actual_test_values, prophet_test_predictions))
                        mape_prophet = mean_absolute_percentage_error(actual_test_values, prophet_test_predictions) * 100
                        
                        # Display metrics
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        with metric_col1:
                            st.metric("MAE", f"{mae_prophet:.2f}%", help="Mean Absolute Error")
                        with metric_col2:
                            st.metric("RMSE", f"{rmse_prophet:.2f}%", help="Root Mean Square Error")
                        with metric_col3:
                            st.metric("MAPE", f"{mape_prophet:.1f}%", help="Mean Absolute Percentage Error")
                        
                        # Create comparison dataframe
                        prophet_comparison = pd.DataFrame({
                            'Year': test_years,
                            'Actual': actual_test_values.round(2),
                            'Predicted': prophet_test_predictions.round(2),
                            'Error': (actual_test_values - prophet_test_predictions).round(2),
                            'Absolute Error': np.abs(actual_test_values - prophet_test_predictions).round(2)
                        })
                        st.dataframe(prophet_comparison, hide_index=True)
                
                # Model Comparison Chart
                st.subheader("📈 Model Performance Comparison")
                
                fig_eval, ax_eval = plt.subplots(figsize=(12, 6))
                
                # Plot actual values
                ax_eval.plot(test_years, actual_test_values, 'o-', label='Actual', color='black', linewidth=2, markersize=8)
                
                # Plot predictions
                if arima_test_predictions is not None:
                    ax_eval.plot(test_years, arima_test_predictions, 's--', label='ARIMA Predictions', color='blue', linewidth=2, markersize=6)
                
                if prophet_test_predictions is not None:
                    ax_eval.plot(test_years, prophet_test_predictions, 'D--', label='Prophet Predictions', color='orange', linewidth=2, markersize=6)
                
                ax_eval.set_xlabel("Year")
                ax_eval.set_ylabel("Inflation Rate (%)")
                ax_eval.set_title(f"Model Predictions vs Actual Values - {selected_country}")
                ax_eval.legend()
                ax_eval.grid(True, alpha=0.3)
                
                st.pyplot(fig_eval)
                
                # Best Model Recommendation
                st.subheader("🏆 Model Recommendation")
                
                if arima_test_predictions is not None and prophet_test_predictions is not None:
                    if mape_arima < mape_prophet:
                        st.success(f"✅ **ARIMA model performs better** with {mape_arima:.1f}% MAPE vs Prophet's {mape_prophet:.1f}% MAPE")
                        st.info("ARIMA is recommended for this dataset based on lower prediction error.")
                    elif mape_prophet < mape_arima:
                        st.success(f"✅ **Prophet model performs better** with {mape_prophet:.1f}% MAPE vs ARIMA's {mape_arima:.1f}% MAPE")
                        st.info("Prophet is recommended for this dataset based on lower prediction error.")
                    else:
                        st.info("Both models perform similarly. Consider using both for ensemble forecasting.")
                elif arima_test_predictions is not None:
                    st.success(f"✅ **ARIMA model** is available with {mape_arima:.1f}% MAPE")
                elif prophet_test_predictions is not None:
                    st.success(f"✅ **Prophet model** is available with {mape_prophet:.1f}% MAPE")
                
                st.markdown("---")
            
            # Plot full forecast comparison
            if arima_forecast is not None or prophet_forecast is not None:
                fig2, ax = plt.subplots(figsize=(12, 5))
                ax.plot(processed['Year'].dt.year, processed['Inflation_Rate'], 'o-', label='Historical', color='gray', linewidth=2)
                
                if arima_forecast is not None:
                    ax.plot(arima_forecast['ds'].dt.year, arima_forecast['yhat'], 's--', label='ARIMA Forecast', color='blue', linewidth=2)
                    ax.fill_between(arima_forecast['ds'].dt.year, arima_forecast['yhat_lower'], arima_forecast['yhat_upper'], alpha=0.2, color='blue')
                
                if prophet_forecast is not None:
                    ax.plot(prophet_forecast['ds'].dt.year, prophet_forecast['yhat'], 'D--', label='Prophet Forecast', color='orange', linewidth=2)
                    ax.fill_between(prophet_forecast['ds'].dt.year, prophet_forecast['yhat_lower'], prophet_forecast['yhat_upper'], alpha=0.2, color='orange')
                
                # Mark train-test split if applicable
                if len(test_data) > 0:
                    split_year = train_data['Year'].dt.year.max()
                    ax.axvline(x=split_year, color='red', linestyle=':', label='Train/Test Split', linewidth=2)
                
                ax.axhline(y=5, color='green', linestyle=':', label='Target (5%)', linewidth=2)
                ax.set_xlabel("Year")
                ax.set_ylabel("Inflation Rate (%)")
                ax.set_title(f"Inflation Forecast - {selected_country}")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig2)
                
                # Download forecasts
                st.subheader("📥 Download Forecasts")
                download_col1, download_col2 = st.columns(2)
                
                if arima_forecast is not None:
                    with download_col1:
                        arima_csv = arima_df.to_csv(index=False)
                        # Clean filename for special characters
                        safe_filename = selected_country.replace(' ', '_').replace('/', '_')
                        st.download_button(
                            label="Download ARIMA Forecast (CSV)",
                            data=arima_csv,
                            file_name=f"{safe_filename}_arima_forecast.csv",
                            mime="text/csv"
                        )
                
                if prophet_forecast is not None:
                    with download_col2:
                        prophet_csv = prophet_df.to_csv(index=False)
                        # Clean filename for special characters
                        safe_filename = selected_country.replace(' ', '_').replace('/', '_')
                        st.download_button(
                            label="Download Prophet Forecast (CSV)",
                            data=prophet_csv,
                            file_name=f"{safe_filename}_prophet_forecast.csv",
                            mime="text/csv"
                        )
    
    except Exception as e:
        st.error(f"Error processing data: {e}")
        st.write("Please check your file format. Expected columns with 'year'/'date' and 'inflation'/'rate'")
