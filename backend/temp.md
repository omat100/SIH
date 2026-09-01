# backend/app.py
from flask import Flask, jsonify, request
from flask_cors import CORS

from src.minesub.config import load_config
from src.minesub.models.gbdt_model import GbdtRiskModel
from src.minesub.models.torch_model import LSTMRiskModel
from src.minesub.predict import predict as ml_predict

app = Flask(__name__)
CORS(app)

_cfg = load_config()
_cfg.ensure_dirs()
_models = {
    "gbdt": GbdtRiskModel.load(_cfg, _cfg.paths["models"] / "gbdt.joblib"),
    "torch": LSTMRiskModel.load(_cfg, _cfg.paths["models"] / "lstm.pt"),
}


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


@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify({"message": "Hello from the Flask backend!"})


@app.route("/api/test")
def test():
    return {
        "message": "Hello from backend",
        "value": 42
    }


@app.route('/api/model/gbdt', methods=['POST'])
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


@app.route('/api/model/torch', methods=['POST'])
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


@app.route('/api/model/health', methods=['GET'])
def model_health():
    return jsonify({
        "models": {
            "gbdt": "loaded" if "gbdt" in _models else "missing",
            "torch": "loaded" if "torch" in _models else "missing"
        },
        "config": str(_cfg.path)
    })


if __name__ == '__main__':
    app.run(port=5000, debug=True)
