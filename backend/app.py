# backend/app.py
import os
import json
import threading
import time
from datetime import datetime
from collections import defaultdict

from flask import Flask, jsonify, request
from flask_cors import CORS

from src.minesub.config import load_config
from src.minesub.models.gbdt_model import GbdtRiskModel
from src.minesub.models.torch_model import LSTMRiskModel
from src.minesub.predict import predict as ml_predict

app = Flask(__name__)
CORS(app)

# --- ESP32 Serial Configuration ---
SERIAL_PORT = os.getenv("SERIAL_PORT", "COM7")
BAUD_RATE = int(os.getenv("BAUD_RATE", "115200"))
SERIAL_TIMEOUT = 1

# --- Global State ---
_cfg = load_config()
_cfg.ensure_dirs()
_models = {
    "gbdt": GbdtRiskModel.load(_cfg, _cfg.paths["models"] / "gbdt.joblib"),
    "torch": LSTMRiskModel.load(_cfg, _cfg.paths["models"] / "lstm.pt"),
}

# Daily buffer: date_str -> list of readings for that day
_daily_buffer = defaultdict(list)
_buffer_lock = threading.Lock()

# Latest predictions
_latest_predictions = {"gbdt": None, "torch": None}
_prediction_lock = threading.Lock()

# Serial connection status
_serial_status = {
    "connected": False,
    "last_error": None,
    "readings_count": 0,
    "buffered_days": 0,
    "last_prediction_ts": None,
}
_status_lock = threading.Lock()


def _update_status(**kwargs):
    with _status_lock:
        _serial_status.update(kwargs)


def _aggregate_daily(readings):
    """Mean of multiple readings per day."""
    n = len(readings)
    return {
        "temp_c": sum(r["temp_c"] for r in readings) / n,
        "humidity_pct": sum(r["humidity_pct"] for r in readings) / n,
        "tilt_deg": sum(r["tilt_deg"] for r in readings) / n,
        "distance_mm": sum(r["distance_mm"] for r in readings) / n,
    }


def _run_inference(daily_readings):
    """Run both models on aggregated daily readings."""
    for model_name in ("gbdt", "torch"):
        try:
            result = ml_predict(daily_readings, model=model_name, cfg=_cfg)
            with _prediction_lock:
                _latest_predictions[model_name] = result
            _update_status(last_prediction_ts=datetime.utcnow().isoformat() + "Z")
        except Exception as e:
            print(f"Inference error ({model_name}): {e}")


def _check_and_run_inference():
    """If >=30 daily readings, run inference for both models."""
    with _buffer_lock:
        sorted_dates = sorted(_daily_buffer.keys())
        if len(sorted_dates) < 30:
            _update_status(buffered_days=len(sorted_dates))
            return
        _update_status(buffered_days=len(sorted_dates))
        recent_dates = sorted_dates[-30:]
        daily_readings = []
        for d in recent_dates:
            agg = _aggregate_daily(_daily_buffer[d])
            agg["date"] = d
            daily_readings.append(agg)
    _run_inference(daily_readings)


def _parse_and_buffer(line):
    """Parse JSON line, validate, buffer by date."""
    try:
        reading = json.loads(line)
        required = {"date", "temp_c", "humidity_pct", "tilt_deg", "distance_mm"}
        if not all(k in reading for k in required):
            return

        dt = datetime.fromisoformat(reading["date"].replace('Z', '+00:00'))
        date_str = dt.date().isoformat()

        with _buffer_lock:
            _daily_buffer[date_str].append({
                "temp_c": float(reading["temp_c"]),
                "humidity_pct": float(reading["humidity_pct"]),
                "tilt_deg": float(reading["tilt_deg"]),
                "distance_mm": float(reading["distance_mm"]),
            })
            total = sum(len(v) for v in _daily_buffer.values())
        _update_status(readings_count=total)

        _check_and_run_inference()

    except (json.JSONDecodeError, ValueError, KeyError):
        pass  # Silently ignore malformed lines


def _serial_reader():
    """Continuous serial reading in background thread."""
    ser = None
    while True:
        try:
            if ser is None or not ser.is_open:
                ser = __import__('serial').Serial(SERIAL_PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT)
                _update_status(connected=True, last_error=None)
                print(f"Serial connected: {SERIAL_PORT} @ {BAUD_RATE}")

            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                if line:
                    _parse_and_buffer(line)

        except Exception as e:
            _update_status(connected=False, last_error=str(e))
            if ser:
                try:
                    ser.close()
                except Exception:
                    pass
                ser = None
            time.sleep(5)  # Retry delay

        time.sleep(0.01)


def _validate_readings(readings):
    if not isinstance(readings, list) or len(readings) < 30:
        return False, "Need >=30 daily readings"
    required = {"date", "temp_c", "humidity_pct", "tilt_deg", "distance_mm"}
    for i, r in enumerate(readings):
        if not isinstance(r, dict):
            return False, f"Reading {i} must be object"
        missing = required - set(r.keys())
        if missing:
            return False, f"Reading {i} missing fields: {sorted(missing)}"
    return True, None


# --- System Endpoints ---
@app.route('/api/system/data', methods=['GET'])
def get_data():
    return jsonify({"message": "Hello from the Flask backend!"})


@app.route("/api/system/test", methods=['GET'])
def test():
    return {
        "message": "Hello from backend",
        "value": 42
    }


# --- Manual Inference Endpoints ---
@app.route('/api/manual/gbdt', methods=['POST'])
def predict_gbdt():
    data = request.get_json(force=True)
    readings = data.get("readings", [])
    valid, err = _validate_readings(readings)
    if not valid:
        return jsonify({"error": err}), 400
    try:
        result = ml_predict(readings, model="gbdt", cfg=_cfg)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/manual/torch', methods=['POST'])
def predict_torch():
    data = request.get_json(force=True)
    readings = data.get("readings", [])
    valid, err = _validate_readings(readings)
    if not valid:
        return jsonify({"error": err}), 400
    try:
        result = ml_predict(readings, model="torch", cfg=_cfg)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/manual/health', methods=['GET'])
def model_health():
    return jsonify({
        "models": {
            "gbdt": "loaded" if "gbdt" in _models else "missing",
            "torch": "loaded" if "torch" in _models else "missing"
        },
        "config": str(_cfg.path)
    })


# --- Live ESP32 Endpoints ---
@app.route('/api/live/status', methods=['GET'])
def esp_status():
    with _status_lock:
        status = dict(_serial_status)
    with _buffer_lock:
        status["buffered_days"] = len(_daily_buffer)
        status["days"] = sorted(_daily_buffer.keys())
    with _prediction_lock:
        status["has_predictions"] = {k: v is not None for k, v in _latest_predictions.items()}
    return jsonify(status)


@app.route('/api/live/latest', methods=['GET'])
def esp_latest():
    with _prediction_lock:
        preds = dict(_latest_predictions)
    return jsonify(preds)


@app.route('/api/live/buffer', methods=['GET'])
def esp_buffer():
    with _buffer_lock:
        return jsonify({
            "days_buffered": len(_daily_buffer),
            "readings_per_day": {d: len(v) for d, v in _daily_buffer.items()},
            "dates": sorted(_daily_buffer.keys()),
        })


@app.route('/api/live/trigger', methods=['POST'])
def esp_trigger():
    """Manually trigger inference on current buffer."""
    with _buffer_lock:
        sorted_dates = sorted(_daily_buffer.keys())
        if len(sorted_dates) < 30:
            return jsonify({"error": f"Need 30 days, have {len(sorted_dates)}"}), 400
        recent_dates = sorted_dates[-30:]
        daily_readings = []
        for d in recent_dates:
            agg = _aggregate_daily(_daily_buffer[d])
            agg["date"] = d
            daily_readings.append(agg)
    _run_inference(daily_readings)
    with _prediction_lock:
        return jsonify(dict(_latest_predictions))


if __name__ == '__main__':
    # Start serial reader thread
    serial_thread = threading.Thread(target=_serial_reader, daemon=True)
    serial_thread.start()
    print(f"Starting ESP32 serial reader on {SERIAL_PORT} @ {BAUD_RATE}")
    app.run(port=5000, debug=True, use_reloader=False)
