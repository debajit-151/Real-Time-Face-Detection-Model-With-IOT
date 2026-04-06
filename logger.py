import csv
import os
from datetime import datetime
import time

LOG_FILE = "logs/recognition_logs.csv"
_last_log_times = {}
COOLDOWN_SECONDS = 10  # Don't log the same person more than once every 10 seconds

def setup_logger():
    if not os.path.exists("logs"):
        os.makedirs("logs")
        
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Name"])

def log_recognition(name):
    if name == "Unknown":
        return  # Usually we don't log "Unknown", or you can change this to log it too
        
    current_time = time.time()
    
    # Check cooldown
    if name in _last_log_times and (current_time - _last_log_times[name]) < COOLDOWN_SECONDS:
        return
        
    # Log the recognition
    _last_log_times[name] = current_time
    timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp_str, name])
        
    print(f"[LOG] Logged recognition for {name} at {timestamp_str}")

# Initialize logger on import
setup_logger()
