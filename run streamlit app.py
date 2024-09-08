import subprocess
import time
import os

# Define the path to the Streamlit app
script_path = 'fittlyf_11.py'

# Start Streamlit server in a new process
process = subprocess.Popen(['streamlit', 'run', script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Give the server a few seconds to start
time.sleep(5)

# Print the URL where the Streamlit app can be accessed
print("Streamlit app is running. Access it at: http://localhost:8501")

# Stream and print the server's output in real-time
try:
    for line in iter(process.stdout.readline, ''):
        print(line, end='rkp5907@gmail.com')
except KeyboardInterrupt:
    print("Streamlit app terminated.")

