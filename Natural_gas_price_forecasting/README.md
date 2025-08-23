# Natural Gas Price Forecasting

A reproducible workflow to forecast major natural gas price benchmarks (Henry Hub, TTF, PSV, JKM) using classical time-series methods and modern neural forecasting tools.

## Contents

- `0_Data_preprocess_eda.ipynb` — data ingestion, cleaning, feature engineering (Fourier features, seasonal averages, event indicators) and export of the cleaned merged CSV used by downstream notebooks.
- `1_EDA_basic_model.ipynb` — exploratory data analysis and classical baselines (ADF/ACF/PACF, ARIMA, SARIMAX, Exponential Smoothing).
- `2_autoArma_prophet.ipynb` — automated ARIMA (pmdarima) and Prophet experiments with changepoints and custom holidays.
- `3_ANN_models.ipynb` — neural forecasting with Darts (N-BEATS, TFT), covariates, scaling and backtesting.
- `4_TimeGPT.ipynb` — Nixtla TimeGPT experiments for multi-series forecasting (requires API key).
- `Streamlit_dashboard.py` — interactive Streamlit dashboard to visualize netback and forecast outputs (expects processed CSVs in `data/processed/`).

## Data

All data are publicly available including the historical spot prices, temperatures, and storage inventory.


## Dashboard demonstration

![dashboard](images/app_video.mp4)