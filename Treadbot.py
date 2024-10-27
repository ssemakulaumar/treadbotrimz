# trading_bot_with_continuous_monitoring.py

from flask import Flask, request, jsonify
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta
import logging
import os
import time

app = Flask(__name__)

# Logging configuration
logging.basicConfig(
    filename='trading_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# Email configuration
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
EMAIL_ADDRESS = 'your_email@example.com'
EMAIL_PASSWORD = 'your_email_password'
RECEIVER_EMAIL = 'your_email@example.com'

# File paths
trade_signal_file = "mt4_trade_signal.json"
performance_log_file = "performance_log.csv"
historical_data_file = "historical_data.csv"

# AI Model Placeholder
models = {}
features = ['close', 'ma_short', 'ma_long', 'rsi']

# Global performance tracking
performance_data = []

# Load or Train AI Model
def train_ai_models():
    global models
    # Fetch historical data using MetaTrader5
    if not mt5.initialize():
        logging.error("MetaTrader 5 initialization failed.")
        return

    # Download historical data for a specific symbol
    symbol = "EURUSD"
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, '2023-01-01', '2023-12-31')
    mt5.shutdown()
    
    if rates is None:
        logging.error("Failed to fetch historical data.")
        return

    # Create DataFrame and preprocess data
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    # Feature engineering: Creating additional indicators
    df['ma_short'] = df['close'].rolling(window=10).mean()
    df['ma_long'] = df['close'].rolling(window=30).mean()
    df['rsi'] = compute_rsi(df['close'], 14)

    # Drop rows with NaN values
    df.dropna(inplace=True)

    # Define features and target
    X = df[features].values
    y = np.where(df['ma_short'].shift(-1) > df['ma_long'].shift(-1), 1, 0)[:-1]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X[:-1], y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train LSTM Model
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, activation='relu', input_shape=(1, X_train_scaled.shape[1])))
    lstm_model.add(Dense(1, activation='sigmoid'))
    lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Reshape input for LSTM
    X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    lstm_model.fit(X_train_lstm, y_train, epochs=100, batch_size=32, verbose=0)

    # Evaluate LSTM Model
    X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
    lstm_pred = (lstm_model.predict(X_test_lstm) > 0.5).astype(int)
    lstm_accuracy = accuracy_score(y_test, lstm_pred)
    models['lstm'] = lstm_model
    logging.info(f"LSTM Model trained with accuracy: {lstm_accuracy * 100:.2f}%")

    # Train Random Forest Model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    models['rf'] = rf_model
    logging.info(f"Random Forest Model trained with accuracy: {rf_accuracy * 100:.2f}%")

def compute_rsi(series, period=14):
    """ Calculate Relative Strength Index (RSI) """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# AI-Based Trading Decision
def make_trade_decision(current_data):
    if not models:
        logging.error("AI models are not trained.")
        return None

    # Prepare the data for prediction
    current_data_scaled = np.array(current_data).reshape(1, -1)
    scaler = StandardScaler().fit(current_data_scaled)
    current_data_scaled = scaler.transform(current_data_scaled)

    # Get predictions from both models
    lstm_prediction = (models['lstm'].predict(current_data_scaled.reshape(1, 1, -1)) > 0.5).astype(int)[0][0]
    rf_prediction = models['rf'].predict(current_data_scaled)[0]

    # Combine predictions using simple majority voting
    final_prediction = np.mean([lstm_prediction, rf_prediction])
    return "buy" if final_prediction > 0.5 else "sell"

# Risk Management Enhancements
def calculate_take_profit(stop_loss, risk_reward_ratio):
    return stop_loss * risk_reward_ratio

def position_size(balance, risk_percentage, stop_loss):
    """ Calculate position size based on account balance and risk percentage """
    risk_amount = balance * risk_percentage / 100
    return risk_amount / stop_loss

@app.route('/trade', methods=['POST'])
def trade():
    data = request.get_json()

    # Extract trade parameters
    symbol = data['symbol']
    lot_size = data['lot_size']
    stop_loss = data['stop_loss']
    risk_reward_ratio = data.get('risk_reward_ratio', 2.0)

    # Get current market data
    market_data = {
        'close': 1.1250,  # Replace with real-time price
        'volume': 1000,  # Replace with actual volume data
        'ma_short': 1.1230,  # Short moving average
        'ma_long': 1.1200,  # Long moving average
        'rsi': 70  # Relative Strength Index
    }

    # Use AI to make a decision
    action = make_trade_decision([market_data['close'], market_data['ma_short'], market_data['ma_long'], market_data['rsi']])
    
    if action is None:
        return jsonify({"status": "error", "message": "AI decision failed"}), 500

    # Calculate dynamic take-profit using risk-reward ratio
    take_profit = calculate_take_profit(stop_loss, risk_reward_ratio)

    # Calculate position size based on account balance and risk
    balance = 10000  # Replace with actual account balance
    size = position_size(balance, 1, stop_loss)  # 1% risk

    # Create the trade signal
    trade_signal = {
        "symbol": symbol,
        "lot_size": size,
        "take_profit": take_profit,
        "stop_loss": stop_loss,
        "action": action
    }

    # Write the trade signal to JSON file
    with open(trade_signal_file, 'w') as f:
        json.dump(trade_signal, f)

    # Log the trade action
    log_entry = f"Trade signal: {trade_signal}"
    logging.info(log_entry)

    # Track performance
    performance_data.append({
        "timestamp": datetime.now(),
        "action": action,
        "trade_signal": trade_signal,
        "success": None  # Placeholder for success indicator
    })

    # Optionally send email notifications
    send_email_notification()

    return jsonify({"status": "success", "message": f"Trade signal {action} sent", "trade_signal": trade_signal})

@app.route('/logout', methods=['POST'])
def logout():
    # Clear the trade signal file
    if os.path.exists(trade_signal_file):
        os.remove(trade_signal_file)
    return jsonify({"status": "success", "message": "Logged out and trade signal cleared"})

def send_email_notification():
    """ Send an email notification with the trading log """
    log_file = 'trading_log.txt'
    with open(log_file, 'r') as f:
        log_content = f.read()

    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = RECEIVER_EMAIL
    msg['Subject'] = 'Trading Log Update'
    msg.attach(MIMEText(log_content, 'plain'))

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        logging.info("Email notification sent successfully.")
    except Exception as e:
        logging.error(f"Failed to send email: {str(e)}")

def monitor_performance():
    """ Continuously monitor and evaluate performance """
    global performance_data
    while True:
        time.sleep(60)  # Check every minute
        if len(performance_data) > 0:
            # Evaluate the last trade success/failure
            last_trade = performance_data[-1]
            # Placeholder: Update success based on some condition (for example, price movement)
            # You can implement logic to check if the last trade was successful
            # e.g. based on market data or feedback from a trade execution system
            last_trade['success'] = True  # Update based on actual performance

            # Log performance
            logging.info(f"Performance Log: {last_trade}")

            # Check if performance falls below a certain threshold
            if len(performance_data) > 10:  # Example threshold
                recent_trades = performance_data[-10:]
                success_rate = sum(1 for trade in recent_trades if trade['success']) / len(recent_trades)
                if success_rate < 0.5:  # Example threshold for performance
                    logging.warning("Performance deterioration detected. Consider retraining models.")
                    train_ai_models()  # Retrain models if performance is low

if __name__ == '__main__':
    train_ai_models()  # Initial model training
    monitor_thread = threading.Thread(target=monitor_performance)
    monitor_thread.start()  # Start performance monitoring in a separate thread
    app.run()
