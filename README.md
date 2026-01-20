# International Stock Price

Stock price prediction using ARIMA, LSTM, and Linear Regression models, integrated with sentiment analysis data. This application provides a user-friendly interface to visualize stock trends and generate predictions for future stock prices.

## Features

*   **Stock Data Visualization:** View historical stock data including Open, High, Low, Close, and Volume.
*   **Multiple Prediction Models:**
    *   **ARIMA (AutoRegressive Integrated Moving Average):** Classic statistical model for time series forecasting.
    *   **LSTM (Long Short-Term Memory):** Recurrent Neural Network (RNN) capable of learning long-term dependencies.
    *   **Linear Regression:** Simple yet effective regression model for trend prediction.
*   **Interactive Interface:** Built with Streamlit for easy interaction and visualization.
*   **Sentiment Analysis Integration:** (Data available in `final_news_sentiment_analysis.csv`)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd stock_price_sentiment_analysis
    ```

2.  **Create and activate a virtual environment (optional but recommended):**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

2.  **Open your browser:** The app will typically run at `http://localhost:8501`.

3.  **Interact with the app:**
    *   Enter a stock ticker symbol (e.g., AAPL, GOOG, NKE).
    *   Select the prediction models you want to run.
    *   View the historical data and prediction results.

## Requirements

*   Python 3.12+
*   See `requirements.txt` for full list of dependencies.
