# src/model_config.py
# Configuration settings for ALL models (fair comparison setup)

# Prophet model hyperparameters (EXACTLY as in notebook)
PROPHET_PARAMS = {
    'changepoint_prior_scale': 0.08,
    'seasonality_prior_scale': 0.5,
    'yearly_seasonality': False,
    'seasonality_mode': 'additive',
    'interval_width': 0.95
}

# ARIMA model search configuration (EXACTLY as in notebook)
ARIMA_PARAMS = {
    'start_p': 0,
    'start_q': 0,
    'max_p': 3,  # Changed from 5 to 3 to match notebook
    'max_q': 3,  # Changed from 5 to 3 to match notebook
    'seasonal': False,
    'stepwise': True,
    'information_criterion': 'aic'
}

# Common configuration used across all models
RANDOM_SEED = 42
TRAIN_SPLIT_YEAR = 2019  # Notebook trains on 1980-2019
FORECAST_START_BUFFER = 1

# Data filtering to match notebook exactly
FILTER_START_YEAR = 1980  # Notebook starts at 1980
FILTER_END_YEAR = 2024    # Notebook ends at 2024