# src/arima_model.py

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from .model_config import (
    ARIMA_PARAMS, 
    RANDOM_SEED, 
    TRAIN_SPLIT_YEAR, 
    FILTER_START_YEAR, 
    FILTER_END_YEAR
)

def run_arima(df, steps=5):
    """
    Run ARIMA model with data filtered to match notebook (1980-2024)
    """
    df = df[['Year', 'Inflation_Rate']].copy()
    df['Year'] = pd.to_datetime(df['Year'])
    
    # CRITICAL: Filter to match notebook's data range (1980-2024)
    df = df[(df['Year'].dt.year >= FILTER_START_YEAR) & 
            (df['Year'].dt.year <= FILTER_END_YEAR)]
    
    df = df.sort_values('Year')
    df.set_index('Year', inplace=True)

    series = df['Inflation_Rate'].astype(float)

    # Train on 1980-2019 (exactly like notebook)
    train = series[series.index.year <= TRAIN_SPLIT_YEAR]

    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)

    auto_model = auto_arima(
        train,
        start_p=ARIMA_PARAMS['start_p'],
        start_q=ARIMA_PARAMS['start_q'],
        max_p=ARIMA_PARAMS['max_p'],
        max_q=ARIMA_PARAMS['max_q'],
        seasonal=ARIMA_PARAMS['seasonal'],
        stepwise=ARIMA_PARAMS['stepwise'],
        information_criterion=ARIMA_PARAMS['information_criterion'],
        suppress_warnings=True,
        random_state=RANDOM_SEED
    )

    order = auto_model.order

    model = ARIMA(train, order=order)
    results = model.fit()

    forecast = results.get_forecast(steps=steps)
    pred = forecast.predicted_mean
    conf = forecast.conf_int()

    # Use FILTER_END_YEAR (2024) for consistent forecasting
    last_year = FILTER_END_YEAR

    forecast_df = pd.DataFrame({
        'ds': pd.date_range(start=f'{last_year+1}-01-01', periods=steps, freq='YS'),
        'yhat': pred.values,
        'yhat_lower': conf.iloc[:, 0].values,
        'yhat_upper': conf.iloc[:, 1].values
    })

    return forecast_df, order