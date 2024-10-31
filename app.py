from asyncio import Event
from datetime import time
from email.mime.nonmultipart import MIMENonMultipart
from email.mime.text import MIMEText
import json
from logging import FileHandler, Handler, basicConfig
import logging
import mailbox
from msilib.schema import File
from multiprocessing.reduction import send_handle
import os
from platform import python_build
import threading
from xml.sax import handler
from flask import Flask
from flask.json import jsonify
from watchdog.events import FileSystemEventHandler
app = Flask(__name__)

@app.route("/Treadbot.py")
# Initialize logging
def basicConfig(filename='trading_log.txt',level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s'):

# Email Configuration
 SMTP_SERVER = 'smtp.gmail.com'  # e.g., 'smtp.gmail.com' for Gmail
SMTP_PORT = 587
EMAIL_ADDRESS = 'umardinaderson@gmail.com'  # Your email address
EMAIL_PASSWORD = 'cristalline0202'    # Your email password or app password if using Gmail
RECEIVER_EMAIL = 'umardinaderson@gmail.com'  # Receiver email (could be the same or different)

# Log File Path
LOG_FILE_PATH = 'trading_log.txt'

# Path to the file that MT4 will read from
TRADE_FILE_PATH = 'mt4_trade_signal.json'

# Watchdog event handler for log changes
class LogFileHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path == LOG_FILE_PATH:
            print(f'{LOG_FILE_PATH} has been modified.')
            self.send_email_notification()


def send_email_notification(self):
        # Read the latest log entry
        with open('r') as log_file:
            log_contents = log_file.read()

             # Create the email
        msg = MIMENonMultipart()
        msg['From'] = ("umardinaderso@gmail.com")
        msg['To'] = ("umardinaderson@gmail.com")
        msg['Subject'] = 'Trading Log Updated'

        # Attach the log contents
        body = f"The trading log has been updated. Here is the latest content:\n\n{log_contents}"
        msg.attach(MIMEText(body, 'plain'))

        # Send the email
        try:
            server = [()]
            server.starttls()  # Start TLS encryption
            server.login()
            text = msg.as_string()
            server.sendmail( text)
            server.quit()
            print("Email sent successfully.")
        except Exception as e:
            print(f"Failed to send email: {str(e)}")

            # Start the watchdog observer in a separate thread
def start_file_observer():
    send_handle (FileHandler)
    observer = observer()
    observer.schedule(Event-handler, path='.', recursive=False)  # Monitor the current directory
    observer.start()
    try:
        while True:
            time.sleep(1)  # Keep the script running
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

    # Flask route to handle trade signals and write them to a JSON file
# Helper function to write trade signals to a JSON file
def write_trade_signal(symbol, lot_size, take_profit, stop_loss, action, trailing_stop, risk_reward_ratio):
    trade_signal = {
        'symbol': symbol,
        'lot_size': lot_size,
        'take_profit': take_profit,
        'stop_loss': stop_loss,
        'action': action,  # "BUY" or "SELL"
        'trailing_stop': trailing_stop,
        'risk_reward_ratio': risk_reward_ratio
    }
    with open("TRADE_FILE_PATH", 'w') as file:
        json.dump(trade_signal, file)

    logging.info(f"Trade signal written: {trade_signal}")
    return True

# Route to handle logout and clean up
@app.route('/logout', methods=['POST'])
def logout():
    # Clear the trade signal file or shut down processes if needed
    if os.path.exists("RADE_FILE_PATH"):
        os.remove("TRADE_FILE_PATH")
    return jsonify({"status": "success", "message": "Logged out successfully"})

if __name__ == '__main__':
    # Start the file observer in a separate thread
    observer_thread = threading.Thread(target=start_file_observer)
    observer_thread.daemon = True  # Ensure thread exits when the main program does
    observer_thread.start()

    # Start Flask server
    app.run(debug=True)
