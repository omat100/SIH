import { useState } from "react";
import ReadingForm from "./ReadingForm";
import EspLivePanel from "./EspLivePanel";

export default function Overview() {
  const [model, setModel] = useState("gbdt");
  const [readingsCount, setReadingsCount] = useState(0);
  const [showResult, setShowResult] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handlePredict = async (readings) => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch(`/api/manual/${model}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ readings }),
      });
      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.error || "Prediction failed");
      }
      const result = await resp.json();
      setPrediction(result);
      setShowResult(true);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadSample = () => {
    const sample = [];
    const baseDate = new Date("2024-01-01");
    for (let i = 0; i < 35; i++) {
      const d = new Date(baseDate);
      d.setDate(d.getDate() + i);
      sample.push({
        date: d.toISOString().split("T")[0],
        temp_c: 18 + Math.sin(i / 5) * 6 + (Math.random() - 0.5) * 2,
        humidity_pct: 55 + Math.cos(i / 5) * 5 + (Math.random() - 0.5) * 5,
        tilt_deg: 0.002 * i + 0.00002 * i * i + (Math.random() - 0.5) * 0.001,
        distance_mm: 0.4 * i + 0.01 * i * i + (Math.random() - 0.5) * 0.5,
      });
    }
    return sample;
  };

  const riskColor = (cls) => ({
    safe: "#22c55e",
    watch: "#f59e0b",
    critical: "#ef4444",
  }[cls] || "#6b7280");

  // regression payload: colour by the fixed geotech band when present,
  // otherwise by the recall-tuned alert flag
  const alertColor = (alert, band) =>
    band ? riskColor(band) : { elevated: "#f59e0b", normal: "#22c55e" }[alert] || "#6b7280";

  const isRegression = (p) =>
    p?.task === "regression" || p?.predicted_excess_rate_mm_day != null;

  return (
    <div className="page-container">
      <div className="page-header">
        <div className="page-header-content">
          <h3 className="page-title">
            <span className="material-symbols-outlined page-icon">memory</span>
            Run Inference
          </h3>
        </div>
      </div>

      <div className="inference-canvas glass-panel">
        {/* Model Selection */}
        <div className="form-section">
          <label className="form-label">Active Model</label>
          <div className="select-wrapper">
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="form-select"
            >
              <option value="gbdt">GBDT</option>
              <option value="torch">LSTM</option>
            </select>
            <span className="select-icon material-symbols-outlined">expand_more</span>
          </div>
        </div>

        {/* Sensor Readings Form */}
        <div className="form-section">
          <div className="section-header">
            <h4 className="section-title">Sensor Readings</h4>
            <div className="readings-badge">
              <span className="material-symbols-outlined">database</span>
              Readings: {readingsCount} / 30
            </div>
          </div>

          <ReadingForm
            model={model}
            loading={loading}
            onPredict={handlePredict}
            onCountChange={setReadingsCount}
          />

          {/* Actions */}
          <div className="form-actions">
            <button
              className="btn-primary"
              onClick={() => {
                const sample = loadSample();
                // We'd need to pass this to ReadingForm - simplified for now
                alert("Load Sample would populate 35 readings");
              }}
              disabled={loading}
            >
              <span className="material-symbols-outlined">upload_file</span>
              Load Sample (35 readings)
            </button>
          </div>
        </div>
      </div>

      {/* Results Section */}
      {showResult && prediction && (
        <div className="results-panel glass-panel">
          <div className="results-header">
            <h3 className="results-title">Prediction Result</h3>
          </div>
          {isRegression(prediction) ? (
            <>
              <div
                className="risk-card"
                style={{ borderColor: alertColor(prediction.alert, prediction.geotech_band) }}
              >
                <div
                  className="risk-class"
                  style={{ background: alertColor(prediction.alert, prediction.geotech_band) }}
                >
                  {(prediction.alert || prediction.geotech_band || "prediction").toUpperCase()}
                </div>
                <div className="meta">
                  <span>Horizon: {prediction.horizon_days} days</span>
                  <span>Model: {prediction.model}</span>
                  <span>As of: {prediction.as_of}</span>
                </div>
              </div>

              <div className="probabilities">
                <h4 className="section-title">Forecast</h4>
                <div className="prob-bar">
                  <span className="prob-label">Excess settlement rate</span>
                  <span className="prob-value">
                    {prediction.predicted_excess_rate_mm_day?.toFixed(4)} mm/day
                  </span>
                </div>
                {prediction.operating_threshold_mm_day != null && (
                  <div className="prob-bar">
                    <span className="prob-label">Alert threshold</span>
                    <span className="prob-value">
                      {prediction.operating_threshold_mm_day.toFixed(4)} mm/day
                    </span>
                  </div>
                )}
                {prediction.geotech_band && (
                  <div className="prob-bar">
                    <span className="prob-label">Geotech band</span>
                    <span className="prob-value">{prediction.geotech_band}</span>
                  </div>
                )}
              </div>
            </>
          ) : (
            <>
              <div className="risk-card" style={{ borderColor: riskColor(prediction.risk_class) }}>
                <div className="risk-class" style={{ background: riskColor(prediction.risk_class) }}>
                  {prediction.risk_class.toUpperCase()}
                </div>
                <div className="meta">
                  <span>Horizon: {prediction.horizon_days} days</span>
                  <span>Model: {prediction.model}</span>
                  <span>As of: {prediction.as_of}</span>
                </div>
              </div>

              <div className="probabilities">
                <h4 className="section-title">Class Probabilities</h4>
                {Object.entries(prediction.probabilities).map(([cls, prob]) => (
                  <div key={cls} className="prob-bar">
                    <span className="prob-label">{cls}</span>
                    <div className="prob-track">
                      <div
                        className="prob-fill"
                        style={{
                          width: `${prob * 100}%`,
                          background: riskColor(cls),
                        }}
                      />
                    </div>
                    <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </>
          )}

          {prediction.signals && Object.keys(prediction.signals).length > 0 && (
            <div className="signals">
              <h4 className="section-title">Key Signals</h4>
              <table className="signals-table">
                <thead>
                  <tr>
                    <th>Signal</th>
                    <th>Value</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(prediction.signals).map(([k, v]) => (
                    <tr key={k}>
                      <td>{k.replace(/_/g, " ")}</td>
                      <td>{typeof v === "number" ? v.toFixed(4) : v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Live Monitoring Panel */}
      <EspLivePanel />
    </div>
  );
}