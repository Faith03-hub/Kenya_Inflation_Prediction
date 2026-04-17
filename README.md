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
```
kenya_inflation_prediction/
├── dashboard.py                 # Main Streamlit application
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── .streamlit/
│   └── config.toml             # Streamlit configuration
├── src/
│   ├── __init__.py             # Package initializer
│   ├── arima_model.py          # ARIMA model implementation
│   ├── prophet_model.py        # Prophet model implementation
│   ├── evaluation.py           # Model evaluation metrics
│   └── model_config.py         # Model configuration parameters
└── pages/
    └── Upload.py               # Data upload page
```

## 🔧 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Local Setup

1. Clone the repository
```bash
git clone https://github.com/your-username/kenya_inflation_prediction.git
cd kenya_inflation_prediction
```

2. Create a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the application
```bash
streamlit run dashboard.py
```

5. Open your browser and navigate to `http://localhost:8501`

## 📦 Dependencies

```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
requests>=2.31.0
plotly>=5.17.0
statsmodels>=0.14.0
pmdarima>=2.0.0
prophet>=1.1.0
scikit-learn>=1.3.0
pandas-datareader>=0.10.0
openpyxl>=3.1.0
```

## 📊 Model Performance

Based on evaluation on 2020-2024 test data:

| Model | MAE | RMSE | Performance |
|-------|-----|------|-------------|
| ARIMA | 3.45% | 4.12% | Good for short-term |
| Prophet | 2.89% | 3.56% | Better overall |

Note: Prophet performed better on Kenya's data due to its ability to handle structural breaks and non-linear trends.

## 🎯 How It Works

### Data Pipeline
1. Data Fetching: Automatically retrieves inflation data from World Bank API
2. Data Cleaning: Handles missing values and standardizes format
3. Exploratory Analysis: Visualizes trends and patterns
4. Model Training: Trains ARIMA and Prophet on 1980-2019 data
5. Evaluation: Tests models on 2020-2024 data
6. Forecasting: Generates predictions for 2025-2029

### Model Selection
- ARIMA: Best for stationary time series with clear autocorrelation patterns
- Prophet: Better for data with trend changes and potential structural breaks

## 📈 Sample Forecast Results (2025-2029)

| Year | ARIMA Forecast | Prophet Forecast | 95% CI Range |
|------|---------------|------------------|---------------|
| 2025 | 10.09% | 5.92% | +-13.7% |
| 2026 | 11.85% | 5.69% | +-13.2% |
| 2027 | 11.85% | 5.46% | +-16.5% |
| 2028 | 11.85% | 5.23% | +-15.4% |
| 2029 | 11.85% | 4.99% | +-14.9% |

##  Usage Guide

### Dashboard Page
- View historical inflation trends
- Compare model performance
- Generate and download forecasts

### Upload Page
1. Click Upload & Forecast in sidebar
2. Upload CSV or Excel file
3. System auto-detects columns
4. Select forecast horizon
5. Click Run Forecast

### Accepted File Formats
- CSV files (any encoding)
- Excel files (.xlsx, .xls)
- KNBS Economic Survey files
- World Bank API exports

### Expected Column Names
- Year column: 'Year', 'year', 'Date', 'date', 'Period', 'period'
- Inflation column: 'Inflation', 'inflation', 'CPI', 'Rate', 'rate'

## 🔍 Key Findings

1. Kenya's inflation has been volatile, peaking at 45.98% in 1993
2. Recent years show stabilization (2024: 4.49%)
3. Prophet model outperforms ARIMA on 2020-2024 test data
4. Wide confidence intervals indicate significant uncertainty in long-term forecasts

## 🚧 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Module not found | Run pip install -r requirements.txt |
| Data not loading | Check internet connection for World Bank API |
| Upload fails | Ensure file has Year and Inflation columns |
| Prophet installation fails | Install pystan first: pip install pystan==2.19.1.1 |

### Streamlit Cloud Deployment

If deploying to Streamlit Cloud:
1. Ensure all dependencies are in requirements.txt
2. No hardcoded local file paths
3. Use relative imports for src modules
4. Set python_version: "3.9" in .streamlit/config.toml

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (git checkout -b feature/AmazingFeature)
3. Commit changes (git commit -m 'Add AmazingFeature')
4. Push to branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- World Bank for providing open data via API
- Facebook Prophet team for the forecasting library
- Statsmodels for ARIMA implementation
- Streamlit for the amazing dashboard framework

## 📧 Contact

Your Name - njengafaith84@gmail.com

Project Link: [https://github.com/your-username/kenya_inflation_prediction](https://github.com/Faith03-hub/Kenya_Inflation_Prediction)

## 📊 Data Source

- Primary Source: World Bank API - Consumer Price Index (FP.CPI.TOTL.ZG)
- Coverage: 1980 - 2024
- Frequency: Annual
- Last Updated: 2024

## 🎯 Future Improvements

- [ ] Add more forecasting models (LSTM, XGBoost)
- [ ] Include economic indicators (GDP, interest rates)
- [ ] Add scenario analysis (optimistic/pessimistic)
- [ ] Implement real-time alerts for threshold breaches
- [ ] Add regional inflation comparison
- [ ] Create API endpoints for programmatic access

---

## ⭐ Star the Project

If you find this project useful, please give it a star on GitHub! It helps others discover the project.

---

Built with ❤️ using Streamlit
```

