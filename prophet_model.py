# src/prophet_model.py

import pandas as pd
import numpy as np
from prophet import Prophet
from .model_config import (
    PROPHET_PARAMS, 
    RANDOM_SEED, 
    TRAIN_SPLIT_YEAR, 
    FILTER_START_YEAR, 
    FILTER_END_YEAR
)

def run_prophet(df, steps=5):
    """
    Run Prophet model with data filtered to match notebook (1980-2024)
    """
    df = df[['Year', 'Inflation_Rate']].copy()
    df['Year'] = pd.to_datetime(df['Year'])
    
    # CRITICAL: Filter to match notebook's data range (1980-2024)
    df = df[(df['Year'].dt.year >= FILTER_START_YEAR) & 
            (df['Year'].dt.year <= FILTER_END_YEAR)]
    
    df = df.sort_values('Year')

    df = df.rename(columns={'Year': 'ds', 'Inflation_Rate': 'y'})
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna()

    # Train on 1980-2019 (exactly like notebook)
    train = df[df['ds'].dt.year <= TRAIN_SPLIT_YEAR]

    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)

    model = Prophet(**PROPHET_PARAMS)
    model.fit(train)

    # Use FILTER_END_YEAR (2024) for consistent forecasting
    last_year = FILTER_END_YEAR

    future = pd.DataFrame({
        'ds': pd.date_range(start=f'{last_year+1}-01-01', periods=steps, freq='YS')
    })

    forecast = model.predict(future)

    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]