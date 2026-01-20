import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime
import math
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import os

# Set page config
st.set_page_config(page_title="International Stock Price Prediction", layout="wide")

st.title("International Stock Price Prediction Web App")

# Sidebar for user input
st.sidebar.header("User Input")
ticker_symbol = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, GOOG, NKE)", "NKE")
selected_models = st.sidebar.multiselect(
    "Select Models to Run",
    ["ARIMA", "LSTM", "Linear Regression"],
    default=["Linear Regression"]
)

# API Key for Alpha Vantage (ideally should be in secrets)
AV_API_KEY = 'I0TWC260RP30RMO5' 

@st.cache_data
def get_historical(quote):
    st.write(f"Fetching data for {quote}...")
    end = datetime.now()
    start = datetime(end.year-2, end.month, end.day)
    
    # Try yfinance first
    data = yf.download(quote, start=start, end=end)
    
    # If yfinance returns empty or Issue, try Alpha Vantage
    if data.empty:
        st.warning(f"yfinance failed to get data for {quote}, trying Alpha Vantage...")
        try:
            ts = TimeSeries(key=AV_API_KEY, output_format='pandas')
            data, meta_data = ts.get_daily_adjusted(symbol=quote, outputsize='full')
            # Formatting to match yfinance structure roughly or just standardizing
            data = data.head(503).iloc[::-1] # Approximate 2 years trading days
            
            # Standardizing column names
            df = pd.DataFrame()
            df['Date'] = data.index
            df['Open'] = data['1. open'].values
            df['High'] = data['2. high'].values
            df['Low'] = data['3. low'].values
            df['Close'] = data['4. close'].values
            df['dVol'] = data['6. volume'].values # different name to avoid issues
            df.index = df['Date']
            return df
        except Exception as e:
            st.error(f"Error fetching data from Alpha Vantage: {e}")
            return pd.DataFrame()
    
    # Process yfinance data
    df = pd.DataFrame(data)
    # yfinance often returns MultiIndex columns if not flattened, depending on version.
    # New yfinance versions might perform differently.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.reset_index(inplace=True)
    return df

def run_arima(df, quote):
    st.subheader("ARIMA Model Predictions")
    
    # Preprocessing for ARIMA
    data_arima = df[['Close']].copy()
    data_arima = data_arima.dropna()
    
    # Plotting Trend
    fig_trend, ax_trend = plt.subplots(figsize=(10, 6))
    ax_trend.plot(data_arima.index, data_arima['Close'])
    ax_trend.set_title(f"{quote} Stock Price Trend")
    st.pyplot(fig_trend)
    
    # Train/Test Split
    quantity = data_arima['Close'].values
    size = int(len(quantity) * 0.80)
    train, test = quantity[0:size], quantity[size:len(quantity)]
    
    history = [x for x in train]
    predictions = list()
    
    progress_bar = st.progress(0)
    
    # ARIMA Loop
    for t in range(len(test)):
        # Update progress
        progress = (t + 1) / len(test)
        progress_bar.progress(progress)
        
        model = ARIMA(history, order=(6, 1, 0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        
    progress_bar.empty()
    
    # Plotting Results
    fig_res, ax_res = plt.subplots(figsize=(10, 6))
    ax_res.plot(test, label='Actual Price')
    ax_res.plot(predictions, label='Predicted Price', color='red')
    ax_res.set_title(f"ARIMA Prediction for {quote}")
    ax_res.legend()
    st.pyplot(fig_res)
    
    arima_pred = predictions[-1] # Next predicted value relative to the test loop end? 
    # Actually, the notebook prints predictions[-2] as "Tomorrow's prediction". 
    # Let's Stick to notebook logic or typical forecasting. 
    # If we want tomorrow's price (future), we need one more step Forecast.
    # The notebook code: arima_pred=predictions[-2] seems slightly off for 'Tomorrow' if testing on past data.
    # However, let's strictly follow notebook logic or make it sensible? 
    # Notebook: predictions[-2]... weird. Let's just output the last prediction from the loop 
    # which corresponds to the last test data point.
    
    # To predict truly 'Tomorrow' (future), we refit on ALL data.
    model_future = ARIMA(quantity, order=(6,1,0))
    model_future_fit = model_future.fit()
    future_forecast = model_future_fit.forecast()
    next_day_pred = future_forecast[0]

    st.success(f"Tomorrow's {quote} Closing Price Prediction by ARIMA: {next_day_pred:.2f}")
    
    error_arima = math.sqrt(mean_squared_error(test, predictions))
    st.write(f"ARIMA RMSE: {error_arima:.4f}")


def run_lstm(df, quote):
    st.subheader("LSTM Model Predictions")
    
    data_lstm = df[['Close']].copy()
    
    # Feature Scaling
    sc = MinMaxScaler(feature_range=(0, 1))
    data_scaled = sc.fit_transform(data_lstm.values.reshape(-1, 1))
    
    # Split
    train_size = int(len(data_scaled) * 0.80)
    train_data = data_scaled[:train_size]
    test_data = data_scaled[train_size:]
    
    def create_dataset(dataset, time_step=7):
        X, Y = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]
            X.append(a)
            Y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(Y)
    
    time_step = 7
    X_train, y_train = create_dataset(train_data, time_step)
    
    # Reshape for LSTM [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    
    # Build Model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(Dropout(0.1))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(50))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    with st.spinner('Training LSTM Model...'):
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        
    # Testing
    # Need inputs for testing from previous days
    # Combine train and test to capture window
    dataset_total = np.concatenate((train_data, test_data), axis=0)
    inputs = dataset_total[len(dataset_total) - len(test_data) - time_step:]
    inputs = inputs.reshape(-1, 1)
    
    X_test = []
    for i in range(time_step, len(inputs)):
        X_test.append(inputs[i-time_step:i, 0])
    X_test = np.array(X_test)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    
    real_stock_price = sc.inverse_transform(test_data)
    
    # Plotting
    fig_lstm, ax_lstm = plt.subplots(figsize=(10, 6))
    ax_lstm.plot(real_stock_price, label='Actual Price')
    ax_lstm.plot(predicted_stock_price, label='Predicted Price')
    ax_lstm.set_title(f"LSTM Prediction for {quote}")
    ax_lstm.legend()
    st.pyplot(fig_lstm)
    
    # Tomorrow's Prediction
    # Use last 7 days of full data
    last_7_days = data_scaled[-time_step:]
    last_7_days = last_7_days.reshape(1, time_step, 1)
    future_pred = model.predict(last_7_days)
    future_pred = sc.inverse_transform(future_pred)
    
    st.success(f"Tomorrow's {quote} Closing Price Prediction by LSTM: {future_pred[0][0]:.2f}")
    
    # RMSE (Trim length to match)
    # Sometimes shapes might mismatch slightly due to windowing, ensure minimal length match
    min_len = min(len(real_stock_price), len(predicted_stock_price))
    error_lstm = math.sqrt(mean_squared_error(real_stock_price[:min_len], predicted_stock_price[:min_len]))
    st.write(f"LSTM RMSE: {error_lstm:.4f}")

def run_lin_reg(df, quote):
    st.subheader("Linear Regression Predictions")
    
    data_lr = df[['Close']].copy()
    data_lr['Close'] = pd.to_numeric(data_lr['Close'], errors='coerce')
    data_lr = data_lr.dropna()
    
    forecast_out = 7
    data_lr['Prediction'] = data_lr['Close'].shift(-forecast_out)
    
    X = np.array(data_lr.drop(['Prediction'], axis=1))
    X = X[:-forecast_out]
    y = np.array(data_lr['Prediction'])
    y = y[:-forecast_out]
    
    # Split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Forecast Input
    X_forecast = np.array(data_lr.drop(['Prediction'], axis=1))[-forecast_out:]
    X_forecast = sc.transform(X_forecast)
    
    # Train
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    
    # Test
    confidence = clf.score(X_test, y_test)
    st.write(f"Linear Regression Confidence: {confidence:.4f}")
    
    y_pred = clf.predict(X_test)
    # The notebook adds a multiplier *(1.04) manually? 
    # "y_test_pred=y_test_pred*(1.04)" in cell 9.
    # I will replicate it if that's the logic intended, though it looks like a manual adjustment.
    y_pred = y_pred * 1.04
    
    # Plotting
    fig_lr, ax_lr = plt.subplots(figsize=(10, 6))
    ax_lr.plot(y_test, label='Actual Price')
    ax_lr.plot(y_pred, label='Predicted Price')
    ax_lr.set_title(f"Linear Regression Prediction for {quote}")
    ax_lr.legend()
    st.pyplot(fig_lr)
    
    error_lr = math.sqrt(mean_squared_error(y_test, y_pred))
    st.write(f"Linear Regression RMSE: {error_lr:.4f}")
    
    # Forecasting
    forecast_set = clf.predict(X_forecast)
    forecast_set = forecast_set * 1.04
    
    # Notebook says "Tomorrow's prediction: lr_pred=forecast_set[0,0] (if reshape) or just first element"
    # Actually forecast_out is 7 days. Ideally we are predicting N days out.
    # We can display the next 7 days or just the first one.
    st.write("Forecast for next 7 days:")
    st.write(forecast_set)
    st.success(f"Tomorrow's {quote} Closing Price Prediction by Linear Regression: {forecast_set[0]:.2f}")


# Main Execution
if ticker_symbol:
    df_data = get_historical(ticker_symbol)
    
    if not df_data.empty:
        st.write(f"Showing recent data for {ticker_symbol}")
        st.dataframe(df_data.tail())
        
        # Plot closing price
        st.line_chart(df_data['Close'])
        
        if st.button("Run Prediction Models"):
            if "ARIMA" in selected_models:
                run_arima(df_data, ticker_symbol)
            if "LSTM" in selected_models:
                run_lstm(df_data, ticker_symbol)
            if "Linear Regression" in selected_models:
                run_lin_reg(df_data, ticker_symbol)
    else:
        st.error("No data found. Please check the ticker symbol.")
else:
    st.info("Please enter a ticker symbol.")
