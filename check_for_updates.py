import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import hashlib

# Email settings
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_USER = "umardinaderson@gmail.com"
EMAIL_PASSWORD = "ycristalline0202"
RECIPIENT_EMAIL = "umardinaderson@gmail.com"
SUBJECT = "Trading Log Updated"

# File settings
FILE_PATH = "/home/umardinanderson/mysite/trading_log.txt"
HASH_FILE = "/home/umardinanderson/mysite/trading_log_hash.txt"

def send_email():
    """Send an email notification."""
    msg = MIMEMultipart()
    msg["From"] = EMAIL_USER
    msg["To"] = RECIPIENT_EMAIL
    msg["Subject"] = SUBJECT

    body = "The trading log file has been updated. Check it at PythonAnywhere."
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_USER, RECIPIENT_EMAIL, msg.as_string())
    print("Email sent.")

def get_file_hash(filepath):
    """Generate a hash for the file content."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def check_for_update():
    """Check if the file content has changed."""
    current_hash = get_file_hash(FILE_PATH)

    try:
        with open(HASH_FILE, "r") as f:
            previous_hash = f.read().strip()
    except FileNotFoundError:
        previous_hash = ""

    if current_hash != previous_hash:
        send_email()
        with open(HASH_FILE, "w") as f:
            f.write(current_hash)
        print("File updated, email sent.")
    else:
        print("No changes detected.")

# Run the check
check_for_update()
