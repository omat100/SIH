from flask import Flask, render_template
from flask_socketio import SocketIO
import serial
import threading
import time

app = Flask(__name__)

socket = SocketIO(app, cors_allowed_origins="*")

try:
    ser = serial.Serial('COM3', 115200)
except Exception as e:
    print("Failed to establish connections")
    ser = None

def read_and_process():
    """Background processing of data"""
    while True:
        socket.emit('connection', {'result': True})
        if ser and ser.in_waiting > 0:
            socket.emit('connection', {'result': True})

        time.sleep(0.02)

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    thread = threading.Thread(target=read_and_process, daemon=True)
    thread.start()

    socket.run(app=app, host='localhost', port=8085)
