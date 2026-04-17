# 📈 Kenya Inflation Rate Prediction Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([https://your-app-url.streamlit.app](https://kenyainflationprediction-cije8oocbvz3rnq6otz5ch.streamlit.app/))
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📊 Overview

This project provides an interactive dashboard for analyzing and forecasting Kenya's inflation rates using two powerful time series models:

- **ARIMA (AutoRegressive Integrated Moving Average)** - Classical statistical model
- **Prophet (Facebook)** - Modern forecasting tool designed for business time series

The dashboard fetches real-time data from the World Bank API and generates forecasts up to 2029, complete with confidence intervals and model performance comparisons.

## ✨ Features

### 🔍 Data Analysis
- **Real-time data fetching** from World Bank API (1980-2024)
- **Interactive visualizations** of historical inflation trends
- **Key metrics** (average, peak, latest inflation rates)
- **Annotated charts** highlighting major economic events

### 🤖 Forecasting Models
- **ARIMA Model** - Automatically tuned using AIC criterion
- **Prophet Model** - Handles trends and seasonality automatically
- **95% Confidence Intervals** for all predictions
- **Model comparison** with MAE and RMSE metrics

### 📤 Upload & Custom Data
- **Upload your own datasets** (CSV, Excel, KNBS files)
- **Auto-detection** of year and inflation columns
- **Support for multiple formats** including World Bank exports

### 📥 Export Capabilities
- **Download forecasts** as CSV files
- **Export historical data** for further analysis

## 🚀 Live Demo

👉 **[Click here to view the live dashboard]([https://your-app-url.streamlit.app](https://kenyainflationprediction-cije8oocbvz3rnq6otz5ch.streamlit.app/))**

## 📸 Screenshots

### Main Dashboard
![Dashboard](screenshots/dashboard.png)

### Forecast Comparison
![Forecast](screenshots/forecast.png)

### Upload Feature
![Upload](screenshots/upload.png)

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Frontend** | Streamlit |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Plotly |
| **Forecasting** | Statsmodels (ARIMA), Prophet |
| **Data Source** | World Bank API |
| **Deployment** | Streamlit Cloud |

## 📁 Project Structure
