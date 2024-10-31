from flask import Flask, request, jsonify
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import numpy as np
from threading import Thread
import MetaTrader5 as mt5  # MetaTrader import
from keras.layers import LSTM
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from datetime import datetime
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

# MetaTrader5 Account Initialization
def initialize_mt5():
    if not mt5.initialize():
        logging.error("MetaTrader5 initialization failed")
        return False
    else:
        account = 12345678  # Replace with your account number
        authorized = mt5.login(account, password="password")  # Replace with your account password
        if not authorized:
            logging.error("Failed to connect to account.")
            return False
    logging.info("MetaTrader5 initialized and connected.")
    return True

# Real-Time Data Fetching
def fetch_real_time_data(symbol="EURUSD"):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logging.error("Failed to retrieve tick data.")
        return None
    return tick

# Trade Execution
def execute_trade(action, symbol="EURUSD", lot=0.1, stop_loss=50, take_profit=100):
    # Calculate price, SL, TP based on action
    tick = fetch_real_time_data(symbol)
    if tick is None:
        return False
    price = tick.ask if action == "buy" else tick.bid
    sl = price - stop_loss * mt5.symbol_info(symbol).point if action == "buy" else price + stop_loss * mt5.symbol_info(symbol).point
    tp = price + take_profit * mt5.symbol_info(symbol).point if action == "buy" else price - take_profit * mt5.symbol_info(symbol).point

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY if action == "buy" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 10,
        "magic": 234000,
        "comment": "AI trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    # Execute order
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error(f"Trade execution failed, code: {result.retcode}")
        return False
    logging.info(f"{action.capitalize()} trade executed for {symbol} at {price}")
    return True

@app.route('/trade', methods=['POST'])
def trade():
    data = request.get_json()
    symbol = data['symbol']
    lot_size = data['lot_size']
    stop_loss = data['stop_loss']
    risk_reward_ratio = data.get('risk_reward_ratio', 2.0)

    # Fetch real-time market data
    tick_data = fetch_real_time_data(symbol)
    if not tick_data:
        return jsonify({"status": "error", "message": "Failed to fetch market data"}), 500

    market_data = {
        'close': tick_data.ask,
        'ma_short': tick_data.ask - 0.0020,
        'ma_long': tick_data.ask - 0.0035,
        'rsi': 70  # Replace with real RSI calculation as needed
    }

    # Make trade decision
    action = make_trade_decision([market_data['close'], market_data['ma_short'], market_data['ma_long'], market_data['rsi']])
    if action is None:
        return jsonify({"status": "error", "message": "AI decision failed"}), 500

    # Calculate take-profit and execute trade
    take_profit = calculate_take_profit(stop_loss, risk_reward_ratio)
    if not execute_trade(action, symbol, lot_size, stop_loss, take_profit):
        return jsonify({"status": "error", "message": "Trade execution failed"}), 500

    return jsonify({"status": "success", "message": f"Trade signal {action} executed", "trade_signal": {"symbol": symbol, "action": action}})

if __name__ == '__main__':
    if initialize_mt5():
        train_ai_models()
        monitor_thread = Thread(target=monitor_performance)
        monitor_thread.start()
        app.run()
    else:
        logging.error("Failed to initialize MetaTrader5")
