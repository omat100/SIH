# backend/app.py
import os
import json
import math
import queue
import re
import threading
import time
from datetime import datetime
from collections import defaultdict

from flask import Flask, Response, jsonify, request, stream_with_context
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

# Number of distinct buffered days required before inference runs. Defaults to 30
# for real field behaviour; lower it via env (or ?min_days= on the trigger route)
# to exercise the pipeline during testing without waiting a month.
MIN_BUFFER_DAYS = int(os.getenv("MIN_BUFFER_DAYS", "30"))

# --- Global State ---
_cfg = load_config()
_cfg.ensure_dirs()
_MODEL_LOADERS = {
    "gbdt": (GbdtRiskModel, "gbdt.joblib"),
    "torch": (LSTMRiskModel, "lstm.pt"),
}
_models = {}
for _name, (_cls, _fname) in _MODEL_LOADERS.items():
    try:
        _models[_name] = _cls.load(_cfg, _cfg.paths["models"] / _fname)
    except Exception as _e:  # artifacts not built yet; health endpoint reports "missing"
        print(f"Model not loaded ({_name}): {_e}")

# Daily buffer: date_str -> list of readings for that day
_daily_buffer = defaultdict(list)
_buffer_lock = threading.Lock()

# Latest predictions
_latest_predictions = {"gbdt": None, "torch": None}
_prediction_lock = threading.Lock()

# Most recent single sensor sample (dict with date + the four metrics), for the
# frontend's live tiles / charts.
_latest_reading = None

# SSE fan-out: one bounded Queue per connected /api/live/stream client.
_subscribers = set()
_subscribers_lock = threading.Lock()

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


def _status_snapshot():
    """Current serial status plus the live buffered-day count."""
    with _status_lock:
        snap = dict(_serial_status)
    with _buffer_lock:
        snap["buffered_days"] = len(_daily_buffer)
    return snap


def _publish(event, data):
    """Fan one SSE event out to every connected /api/live/stream client."""
    with _subscribers_lock:
        subs = list(_subscribers)
    for q in subs:
        try:
            q.put_nowait((event, data))
        except queue.Full:
            pass  # slow client; drop this event for it


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


def _check_and_run_inference(min_days=None):
    """If enough distinct days are buffered, run inference for both models."""
    if min_days is None:
        min_days = MIN_BUFFER_DAYS
    with _buffer_lock:
        sorted_dates = sorted(_daily_buffer.keys())
        _update_status(buffered_days=len(sorted_dates))
        if len(sorted_dates) < min_days:
            return
        recent_dates = sorted_dates[-min_days:]
        daily_readings = []
        for d in recent_dates:
            agg = _aggregate_daily(_daily_buffer[d])
            agg["date"] = d
            daily_readings.append(agg)
    _run_inference(daily_readings)
    with _prediction_lock:
        preds = dict(_latest_predictions)
    _publish("prediction", preds)


# --- ASCII console-block parsing --------------------------------------------
# The receiver prints human-readable packets like:
#     ==== SENSOR DATA RECEIVED ====
#     Temperature: 28.9 °C
#     Humidity: 26.1 %
#     Accel X: -0.08 g
#     ...
#     ==============================
# We reassemble those packets and pull the numbers out of them.

# "Label: <number><unit>" -> capture the label and only the leading number, so
# trailing units (°C, %, g, °/s, cm) are ignored. Text-only values ("Status: OK")
# and banner/separator lines do not match.
_CV_FIELD_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_ ]*?)\s*:\s*([-+]?\d*\.?\d+)")

# Labels pulled from a single packet, matched case-insensitively.
_CONSOLE_FIELDS = ("temperature", "humidity", "accel x", "accel y", "accel z", "distance")


def _parse_cv_field(line):
    """Parse one 'Label: value' console line -> (label, value_float) or None."""
    m = _CV_FIELD_RE.match(line)
    if not m:
        return None
    return m.group(1).strip(), float(m.group(2))


def _extract_console_reading(block_lines):
    """Build a sensor sample from one console packet's lines.

    Returns ``{"temp_c", "humidity_pct", "tilt_deg", "distance_mm"}`` or ``None``
    when a required field is missing (incomplete packet). No timestamp is added.
    """
    vals = {}
    for line in block_lines:
        parsed = _parse_cv_field(line)
        if parsed is None:
            continue
        label, value = parsed
        key = label.lower()
        if key in _CONSOLE_FIELDS:
            vals[key] = value

    if any(k not in vals for k in _CONSOLE_FIELDS):
        return None

    ax, ay, az = vals["accel x"], vals["accel y"], vals["accel z"]
    tilt_deg = math.degrees(math.atan2(math.sqrt(ax ** 2 + ay ** 2), az))
    return {
        "temp_c": vals["temperature"],
        "humidity_pct": vals["humidity"],
        "tilt_deg": tilt_deg,
        "distance_mm": vals["distance"] * 10.0,  # console prints centimetres
    }


def _buffer_reading(sample):
    """Append one dated sample to the daily buffer and maybe run inference.

    ``sample`` must carry ``date`` (ISO string) plus the four sensor metrics.
    """
    global _latest_reading

    dt = datetime.fromisoformat(sample["date"].replace('Z', '+00:00'))
    date_str = dt.date().isoformat()

    metrics = {
        "temp_c": float(sample["temp_c"]),
        "humidity_pct": float(sample["humidity_pct"]),
        "tilt_deg": float(sample["tilt_deg"]),
        "distance_mm": float(sample["distance_mm"]),
    }

    with _buffer_lock:
        _daily_buffer[date_str].append(dict(metrics))
        total = sum(len(v) for v in _daily_buffer.values())
        buffered_days = len(_daily_buffer)
        _latest_reading = {"date": sample["date"], **metrics}

    _update_status(readings_count=total, buffered_days=buffered_days)
    _publish("reading", {**_latest_reading, "readings_count": total})

    _check_and_run_inference()
    _publish("status", _status_snapshot())


def _parse_and_buffer(line):
    """Legacy JSON path: parse one JSON line, validate, buffer by date.

    Returns ``True`` if the line was a usable JSON reading, ``False`` otherwise
    (so the caller can fall back to the ASCII console-block parser).
    """
    try:
        reading = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return False

    required = {"date", "temp_c", "humidity_pct", "tilt_deg", "distance_mm"}
    if not isinstance(reading, dict) or not all(k in reading for k in required):
        return False

    try:
        _buffer_reading(reading)
    except (ValueError, KeyError):
        return False
    return True


def _is_block_boundary(line):
    """True for a banner/separator line that delimits one console packet."""
    stripped = line.strip()
    if len(stripped) >= 3 and set(stripped) == {"="}:
        return True
    return "SENSOR DATA RECEIVED" in stripped.upper()


def _handle_console_line(line, block):
    """Accumulate ``line`` into ``block``; on a boundary, flush a complete packet.

    Returns the (possibly reset) block list. Complete packets are stamped with
    the current UTC time and pushed into the daily buffer.
    """
    if _is_block_boundary(line):
        if block:
            sample = _extract_console_reading(block)
            if sample is not None:
                sample["date"] = datetime.utcnow().isoformat() + "Z"
                _buffer_reading(sample)
        return []
    block.append(line)
    return block


def _serial_reader():
    """Continuous serial reading in background thread."""
    ser = None
    block = []
    while True:
        try:
            if ser is None or not ser.is_open:
                ser = __import__('serial').Serial(SERIAL_PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT)
                _update_status(connected=True, last_error=None)
                print(f"Serial connected: {SERIAL_PORT} @ {BAUD_RATE}")

            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='replace').strip()
                if line:
                    # Try the legacy JSON path first; fall back to accumulating
                    # human-readable console packets.
                    if not _parse_and_buffer(line):
                        block = _handle_console_line(line, block)

        except Exception as e:
            block = []
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


@app.route('/api/live/readings', methods=['GET'])
def esp_readings():
    """One-shot snapshot the frontend uses to hydrate: status, latest reading,
    full per-day buffer, counts and predictions."""
    with _buffer_lock:
        buffer = {d: [dict(r) for r in v] for d, v in _daily_buffer.items()}
    with _prediction_lock:
        predictions = dict(_latest_predictions)
    readings_count = sum(len(v) for v in buffer.values())
    return jsonify({
        "status": _status_snapshot(),
        "latest_reading": _latest_reading,
        "readings_count": readings_count,
        "buffered_days": len(buffer),
        "predictions": predictions,
        "buffer": buffer,
    })


@app.route('/api/live/stream', methods=['GET'])
def esp_stream():
    """Server-Sent Events: 'reading', 'prediction' and 'status' events as they
    happen. Clients connect with EventSource('/api/live/stream')."""
    def gen():
        q = queue.Queue(maxsize=200)
        with _subscribers_lock:
            _subscribers.add(q)
        try:
            # Prime the client with the current state.
            yield f"event: status\ndata: {json.dumps(_status_snapshot())}\n\n"
            with _prediction_lock:
                preds = dict(_latest_predictions)
            yield f"event: prediction\ndata: {json.dumps(preds)}\n\n"
            while True:
                try:
                    event, data = q.get(timeout=15)
                    yield f"event: {event}\ndata: {json.dumps(data)}\n\n"
                except queue.Empty:
                    yield ": keep-alive\n\n"
        finally:
            with _subscribers_lock:
                _subscribers.discard(q)

    return Response(
        stream_with_context(gen()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route('/api/live/trigger', methods=['POST'])
def esp_trigger():
    """Manually trigger inference on current buffer.

    Accepts ``?min_days=N`` to force inference on fewer days for testing;
    defaults to ``MIN_BUFFER_DAYS``.
    """
    try:
        min_days = int(request.args.get("min_days", MIN_BUFFER_DAYS))
    except (TypeError, ValueError):
        return jsonify({"error": "min_days must be an integer"}), 400
    if min_days < 1:
        return jsonify({"error": "min_days must be >= 1"}), 400
    with _buffer_lock:
        sorted_dates = sorted(_daily_buffer.keys())
        if len(sorted_dates) < min_days:
            return jsonify({"error": f"Need {min_days} days, have {len(sorted_dates)}"}), 400
        recent_dates = sorted_dates[-min_days:]
        daily_readings = []
        for d in recent_dates:
            agg = _aggregate_daily(_daily_buffer[d])
            agg["date"] = d
            daily_readings.append(agg)
    _run_inference(daily_readings)
    with _prediction_lock:
        preds = dict(_latest_predictions)
    _publish("prediction", preds)
    return jsonify(preds)


if __name__ == '__main__':
    # Start serial reader thread
    serial_thread = threading.Thread(target=_serial_reader, daemon=True)
    serial_thread.start()
    print(f"Starting ESP32 serial reader on {SERIAL_PORT} @ {BAUD_RATE}")
    app.run(port=5000, debug=True, use_reloader=False, threaded=True)
